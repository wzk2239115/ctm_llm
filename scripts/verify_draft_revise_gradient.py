"""Functional check: draft-revise CE now carries gradient (was inert before fix).

Before fix (draft_pred = current_prediction.detach()): dp.requires_grad=False,
backward gives no param grads -> CE term trained nothing.
After  fix (draft_pred = current_prediction):         dp.requires_grad=True,
backward updates params -> CE term is real deep supervision.

Run: python scripts/verify_draft_revise_gradient.py
"""
import sys, torch
sys.path.insert(0, ".")
from baseline.models.ctm import ContinuousThoughtMachine
from baseline.data.custom_datasets import ParityDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

model = ContinuousThoughtMachine(
    iterations=10, d_model=128, d_input=64, heads=4,
    n_synch_out=32, n_synch_action=32, synapse_depth=1,
    memory_length=8, deep_nlms=True, memory_hidden_dims=128,
    do_layernorm_nlm=False, backbone_type="parity_backbone",
    positional_embedding_type="custom-rotational-1d",
    out_dims=128, prediction_reshaper=[64, 2],
    neuron_select_type="random",
).to(dev)
model.train()
model.draft_mode = "revise"
model.draft_block_size = 2          # boundary at tick 1
model.draft_corrupt_prob = 1.0      # force corruption so revise path fully exercised

# one real parity batch
data = ParityDataset(sequence_length=64, length=64)
inputs = data[0][0].unsqueeze(0).to(dev)    # (1, 64)

out = model(inputs)
extras = out[3] if isinstance(out, tuple) and len(out) == 4 else {}
assert "draft_prediction" in extras, "draft_prediction not produced (boundary not hit?)"
dp = extras["draft_prediction"]

print(f"draft_prediction: shape={tuple(dp.shape)} requires_grad={dp.requires_grad} "
      f"grad_fn={dp.grad_fn}")

# zero grads, backward through draft CE only
model.zero_grad(set_to_none=True)
loss = dp.float().sum()              # any differentiable scalar; exercises full graph
loss.backward()

n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for p in model.parameters())
print(f"params with non-zero grad: {n_with_grad}/{n_total}")

ok = dp.requires_grad and n_with_grad > 0
print("RESULT:", "PASS — draft CE now has gradient (deep supervision active)" if ok
      else "FAIL — still inert (detach not removed?)")
sys.exit(0 if ok else 1)
