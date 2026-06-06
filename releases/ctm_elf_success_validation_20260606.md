# CTM+ELF Training Validation

Date: 2026-06-06

This version records the first successful CTM+ELF training validation run. The
goal of this release is to prove that the current CTM-LLM training path,
multi-node infra, checkpointing, and CTM tick supervision can run stably and
converge on the available SFT corpus.

This is not a language-quality release. Chat eval still shows semantic drift and
weak instruction following, so the next quality bottleneck is data/alignment
rather than basic trainability.

## Validation Scope

- Architecture: CTM-LLM built from CTM-style internal iteration and ELF-inspired
  causal language modeling path.
- Tokenizer: minimind-o compatible tokenizer, `vocab_size=6400`.
- Data: `dataset_data/sft_t2a_mini.parquet`, about 515k English dialog rows.
- Logging: TensorBoard by default. SwanLab is optional.
- Infra: no-SSH `ctmctl` pool on two shared-storage nodes.
- Hardware: two nodes, 16 total NVIDIA H800/H100-class 80GB GPUs.

## Cluster

Pool config:

```bash
infra/clusters/h100_2nodes.env
```

Nodes used:

```text
11.131.210.78  rank=0  master
11.131.210.3   rank=1  worker
```

The two machines mount the same repo/data storage, so `SHARED_REPO=1` is enabled
and pool workers serialize Git updates through `.ctm_pool/git_update.lock`.

## Training Command

The successful long validation run was submitted from the master node:

```bash
./scripts/ctmctl pool submit infra/clusters/h100_2nodes.env \
  --swanlab_name ctm-2node-1024-16l-bs16-44ep \
  --save_weight ctm_2node_1024_16l_bs16_44ep \
  --epochs 44 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --max_seq_len 512 \
  --num_hidden_layers 16 \
  --hidden_size 1024 \
  --d_model 512 \
  --d_input 256 \
  --heads 8 \
  --n_synch_out 512 \
  --n_synch_action 512 \
  --iterations 4 \
  --memory_length 5 \
  --synapse_depth 2 \
  --num_workers 4 \
  --log_interval 20 \
  --save_interval 1000
```

## Result

Final observed training log:

```text
Epoch[44/44](2013/2013) | loss:1.7685 lr:5.00e-05 | best_tick:1.4 conf_tick:1.2 | elapsed:25.0min epoch_eta:0s | total:25.0min(100.0%)
Checkpoint saved: out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth
Epoch 44 done in 25.0min
Training complete! Total: 18.35h
```

TensorBoard run directory on the compute machine:

```bash
/home/jovyan/h800fast/wangzekai/ctm_llm/runs/ctm-2node-1024-16l-bs16-44ep
```

Main checkpoint:

```bash
out/ctm_2node_1024_16l_bs16_44ep_1024.pth
```

Resume checkpoint:

```bash
out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth
```

The resume checkpoint is expected to be much larger than the inference
checkpoint because it includes optimizer/scaler state.

## Success Criteria

This run passes the training-mechanics validation because:

- Two-node `ctmctl` pool startup, worker registration, task ack, and launch path
  worked on shared storage.
- Distributed training completed the full 44-epoch schedule without NCCL or
  rendezvous failure.
- TensorBoard logs showed a sustained downward loss curve.
- Training loss reached about `1.77` after `18.35h`.
- Checkpoint save/resume artifacts were produced successfully.
- CTM tick metrics (`best_tick`, `conf_tick`) stayed live during training, so the
  internal-clock path is active and measurable.

## Known Limits

- Qualitative `eval.py` generations are not yet reliable for semantic accuracy
  or instruction following.
- The current SFT corpus is English-only and appears to encourage surface-form
  distribution fitting more than grounded instruction behavior.
- This run should be treated as evidence that the model can train and fit the
  corpus, not as evidence that it is already a useful assistant.
- CTM internal-clock usefulness still needs targeted visualization and ablation.

## Recommended Follow-up Checks

Run loss/perplexity evaluation on held-out subsets:

```bash
python eval_loss.py \
  --weight out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth \
  --hidden_size 1024 \
  --num_hidden_layers 16 \
  --d_model 512 \
  --d_input 256 \
  --heads 8 \
  --n_synch_out 512 \
  --n_synch_action 512 \
  --iterations 4 \
  --memory_length 5 \
  --synapse_depth 2 \
  --max_seq_len 512 \
  --samples 4096 \
  --split random
```

Run qualitative generation with conservative decoding:

```bash
python eval.py \
  --weight out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth \
  --hidden_size 1024 \
  --num_hidden_layers 16 \
  --d_model 512 \
  --d_input 256 \
  --heads 8 \
  --n_synch_out 512 \
  --n_synch_action 512 \
  --iterations 4 \
  --memory_length 5 \
  --synapse_depth 2 \
  --temperature 0.3 \
  --top_p 0.8 \
  --top_k 40 \
  --repetition_penalty 1.08 \
  --prompt "Explain what a neural network is in simple words." \
  --max_new_tokens 160
```

Next engineering target:

- Build a CTM clock probe UI/server that keeps the model on the compute machine
  and visualizes per-token tick choice, certainty, entropy, and generated text.
- Add ablations for `iterations`, `memory_length`, `cross_layer_state`, and
  `self_cond` to separate CTM-clock signal from ordinary capacity gain.
