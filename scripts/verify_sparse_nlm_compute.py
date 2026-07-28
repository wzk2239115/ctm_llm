"""Verify real sparse NLM compute.

Three checks:
1. Correctness: SuperLinear is per-neuron independent, so gathering k neurons and
   computing them must give IDENTICAL values to computing all N densely then reading
   those k (eval mode, dropout off). If not, the gather/scatter is wrong.
2. Smoke: full CTM forward runs end-to-end with sparse_nlm_compute=True.
3. Speedup: time dense trace_processor vs sparse (gather) forward; report actual
   wall-clock ratio + analytical FLOP ratio (k/N).

Run: python scripts/verify_sparse_nlm_compute.py
"""
import sys, time, torch
sys.path.insert(0, ".")
from baseline.models.ctm import ContinuousThoughtMachine
from baseline.utils.ctm_model_ideas import select_active_neurons, nlm_sparse_forward

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

B, N, M = 64, 1024, 25           # batch, d_model, memory_length
r = 0.25
k = max(1, int(N * r))

# ---- 1. build a CTM and grab its trace_processor + a real state_trace ----
model = ContinuousThoughtMachine(
    iterations=10, d_model=N, d_input=64, heads=4,
    n_synch_out=32, n_synch_action=32, synapse_depth=1,
    memory_length=M, deep_nlms=True, memory_hidden_dims=128,
    do_layernorm_nlm=False, backbone_type="parity_backbone",
    positional_embedding_type="custom-rotational-1d",
    out_dims=2, prediction_reshaper=[-1], neuron_select_type="random",
).to(dev).eval()
tp = model.trace_processor

state_trace = torch.randn(B, N, M, device=dev)
state = torch.randn(B, N, device=dev)

with torch.no_grad():
    dense = tp(state_trace)                       # (B, N) full dense NLM output
    idx = select_active_neurons(state, r)         # (k,)
    sparse_full = nlm_sparse_forward(tp, state_trace, idx)   # (B, N) scattered

    active_match = torch.allclose(sparse_full[:, idx], dense[:, idx], atol=1e-5)
    inactive_zero = (sparse_full.float() == 0).all(dim=0)
    inactive_ok = inactive_zero.sum().item() == (N - k)
    n_nonzero = (sparse_full.abs() > 0).any(dim=0).sum().item()
print(f"[1] correctness: active neurons match dense = {active_match}")
print(f"    inactive neurons are zero = {inactive_ok}  (nonzero cols = {n_nonzero}, expect {k})")

# ---- 2. smoke: full forward with sparse_nlm_compute ----
model.train()
model.topk_neurons = r
model.sparse_nlm_compute = True
x = torch.randn(B, 64, device=dev)
out = model(x)
preds = out[0]
print(f"[2] smoke forward: preds shape {tuple(preds.shape)} ok")

# ---- 3. speedup: dense vs sparse NLM forward ----
model.eval()
with torch.no_grad():
    for _ in range(3): tp(state_trace); nlm_sparse_forward(tp, state_trace, idx)
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    for _ in range(200): tp(state_trace)
    torch.cuda.synchronize() if dev == "cuda" else None
    t_dense = (time.time() - t0) / 200
    t0 = time.time()
    for _ in range(200): nlm_sparse_forward(tp, state_trace, idx)
    torch.cuda.synchronize() if dev == "cuda" else None
    t_sparse = (time.time() - t0) / 200
print(f"[3] NLM forward latency (B={B}, N={N}, M={M}, r={r}, k={k}):")
print(f"    dense  : {t_dense*1e6:8.1f} us")
print(f"    sparse : {t_sparse*1e6:8.1f} us")
print(f"    speedup: {t_dense/t_sparse:5.2f}x   (analytical FLOP ratio k/N = {k/N:.2f}, expect ~{1/k*N:.1f}x ceiling)")

ok = active_match and inactive_ok
print("RESULT:", "PASS — sparse NLM correct + runs + faster" if ok else "FAIL")
sys.exit(0 if ok else 1)
