"""Step 2 functional divergence: after the detach fix, does the draft CE term
actually change the training trajectory? (Before fix it was inert -> identical.)

Isolates the CE effect: corrupt_prob=0 for both (no noise), only draft_revise_weight
differs. Same seed/init/data. If params diverge -> CE is now active.

Also reports the shape-guard status (parity skips CE: dp (B,128) vs targets (B,64)).

Run: python scripts/verify_draft_revise_divergence.py
"""
import sys, torch
sys.path.insert(0, ".")
from baseline.models.ctm import ContinuousThoughtMachine

dev = "cuda" if torch.cuda.is_available() else "cpu"


def build_and_run(w, seed=0, steps=60):
    torch.manual_seed(seed)
    model = ContinuousThoughtMachine(
        iterations=10, d_model=128, d_input=64, heads=4,
        n_synch_out=32, n_synch_action=32, synapse_depth=1,
        memory_length=8, deep_nlms=True, memory_hidden_dims=128,
        do_layernorm_nlm=False, backbone_type="parity_backbone",
        positional_embedding_type="custom-rotational-1d",
        out_dims=2, prediction_reshaper=[-1],   # 2-class -> CE size-guard passes
        neuron_select_type="random",
    ).to(dev)
    model.train()
    model.draft_mode = "revise"
    model.draft_block_size = 2
    model.draft_corrupt_prob = 0.0            # isolate CE: no noise
    model.draft_revise_weight = w

    torch.manual_seed(seed + 100)             # data order fixed
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B = 16
    losses = []
    for step in range(steps):
        x = torch.randn(B, 64, device=dev)
        t = torch.randint(0, 2, (B,), device=dev)
        out = model(x)
        extras = out[3] if isinstance(out, tuple) and len(out) == 4 else {}
        pred = out[0]                         # (B, 2, T)
        # main loss: CE at final tick
        loss_main = torch.nn.functional.cross_entropy(pred[:, :, -1], t)
        loss = loss_main
        guard_pass = False
        if "draft_prediction" in extras:
            dp = extras["draft_prediction"]   # (B, 2)
            dp_flat = dp.reshape(-1, dp.size(-1))
            tgt_flat = t.reshape(-1)
            if dp_flat.size(0) == tgt_flat.size(0):
                guard_pass = True
                loss = loss + w * torch.nn.functional.cross_entropy(dp_flat, tgt_flat.long())
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    # param checksum
    cksum = sum(p.detach().float().sum().item() for p in model.parameters())
    return cksum, losses, guard_pass


if __name__ == "__main__":
    ck0, l0, g0 = build_and_run(w=0.0)
    ck1, l1, g1 = build_and_run(w=0.1)
    print(f"shape guard passes (CE actually added): {g0}")
    print(f"w=0.0 param checksum: {ck0:.4f}  final loss {l0[-1]:.4f}")
    print(f"w=0.1 param checksum: {ck1:.4f}  final loss {l1[-1]:.4f}")
    diverge = abs(ck0 - ck1) > 1e-6
    print(f"params diverged: {diverge}")
    print("RESULT:", "PASS — draft CE now changes training (was inert before fix)" if (diverge and g0)
          else "FAIL — CE still inert or guard blocks it")
    sys.exit(0 if (diverge and g0) else 1)
