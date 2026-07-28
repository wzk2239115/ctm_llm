"""Step 2 functional divergence for sparse NLM compute.

Same seed/init/data, only sparse_nlm_compute differs (True=real gather vs
False=legacy post-hoc mask). Different active-set selection basis (pre-NLM
from `state` vs post-NLM from `activated_state`) => params must diverge.
If identical, the flag is wired wrong (inert).

Run: python scripts/verify_sparse_nlm_divergence.py
"""
import sys, torch
sys.path.insert(0, ".")
from baseline.models.ctm import ContinuousThoughtMachine

dev = "cuda" if torch.cuda.is_available() else "cpu"


def run(sparse, seed=0, steps=50):
    torch.manual_seed(seed)
    model = ContinuousThoughtMachine(
        iterations=10, d_model=128, d_input=64, heads=4,
        n_synch_out=32, n_synch_action=32, synapse_depth=1,
        memory_length=8, deep_nlms=True, memory_hidden_dims=128,
        do_layernorm_nlm=False, backbone_type="parity_backbone",
        positional_embedding_type="custom-rotational-1d",
        out_dims=2, prediction_reshaper=[-1], neuron_select_type="random",
    ).to(dev)
    model.train()
    model.topk_neurons = 0.25
    model.sparse_nlm_compute = sparse
    torch.manual_seed(seed + 100)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B = 32
    for _ in range(steps):
        x = torch.randn(B, 64, device=dev)
        t = torch.randint(0, 2, (B,), device=dev)
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out[0][:, :, -1], t)
        opt.zero_grad(); loss.backward(); opt.step()
    return sum(p.detach().float().sum().item() for p in model.parameters())


if __name__ == "__main__":
    ck_post = run(sparse=False)   # legacy post-hoc mask
    ck_real = run(sparse=True)    # real sparse NLM compute
    diverge = abs(ck_post - ck_real) > 1e-6
    print(f"post-hoc mask checksum: {ck_post:.4f}")
    print(f"real sparse   checksum: {ck_real:.4f}")
    print(f"diverged: {diverge}")
    print("RESULT:", "PASS — sparse_nlm_compute changes training (flag is active)"
          if diverge else "FAIL — flag inert")
    sys.exit(0 if diverge else 1)
