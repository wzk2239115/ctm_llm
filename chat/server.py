import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, '..'))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_service import CTMChatService

app = FastAPI(title='CTM-LLM Chat')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

service: CTMChatService = None


class ChatRequest(BaseModel):
    messages: list
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.08
    confidence_threshold: float = 0.8


@app.get('/')
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'index.html'))


@app.post('/api/chat')
async def chat(req: ChatRequest):
    def stream():
        for chunk in service.generate_stream(
            req.messages,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            confidence_threshold=req.confidence_threshold,
        ):
            yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
    return StreamingResponse(stream(), media_type='text/event-stream')


@app.get('/api/model_info')
async def model_info():
    if service is None:
        return {'status': 'not_loaded'}
    cfg = service.config
    return {
        'status': 'loaded',
        'hidden_size': cfg.hidden_size,
        'num_hidden_layers': cfg.num_hidden_layers,
        'd_model': cfg.d_model,
        'd_input': cfg.d_input,
        'iterations': service.iterations,
        'memory_length': cfg.memory_length,
        'heads': cfg.heads,
        'n_synch_out': cfg.n_synch_out,
        'n_synch_action': cfg.n_synch_action,
        'synapse_depth': cfg.synapse_depth,
        'vocab_size': cfg.vocab_size,
    }


def parse_args():
    p = argparse.ArgumentParser(description='CTM-LLM Chat WebUI')
    p.add_argument('--weight', type=str,
                   default='out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth')
    p.add_argument('--tokenizer_path', type=str, default='./model_tokenizer')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--host', type=str, default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--hidden_size', type=int, default=1024)
    p.add_argument('--num_hidden_layers', type=int, default=16)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--d_input', type=int, default=256)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--memory_length', type=int, default=5)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--n_synch_out', type=int, default=512)
    p.add_argument('--n_synch_action', type=int, default=512)
    p.add_argument('--synapse_depth', type=int, default=2)
    p.add_argument('--self_cond', type=int, default=1)
    p.add_argument('--cross_layer_state', type=int, default=1)
    p.add_argument('--block_size', type=int, default=4)
    p.add_argument('--num_iters', type=int, default=None)
    return p.parse_args()


def main():
    global service
    args = parse_args()

    service = CTMChatService(
        weight_path=args.weight,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        d_model=args.d_model,
        d_input=args.d_input,
        iterations=args.iterations,
        memory_length=args.memory_length,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        self_cond=bool(args.self_cond),
        cross_layer_state=bool(args.cross_layer_state),
        block_size=args.block_size,
        num_iters=args.num_iters,
    )

    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app.mount('/static', StaticFiles(directory=static_dir), name='static')

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
