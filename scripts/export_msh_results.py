#!/usr/bin/env python3
"""Export experiment results from checkpoints to CSV."""

import torch, csv, os, sys, glob

LOG_DIRS = [
    "logs/dtt/dtt01",
    "logs/msh/msh08",
    "logs/msh/msh01",
    "logs/lg",
]

def extract_experiment(log_dir):
    """Extract metrics from a single experiment directory."""
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")
    args_path = os.path.join(log_dir, "args.txt")

    if not os.path.exists(ckpt_path):
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"name": os.path.basename(log_dir), "error": str(e)}

    # Parse args
    config = {}
    if os.path.exists(args_path):
        with open(args_path) as f:
            args_text = f.read()
        for token in args_text.split():
            if "=" in token and not token.startswith("-"):
                k, v = token.split("=", 1)
                config[k] = v.strip("'\"")

    ta = ckpt.get("test_accuracies", [])
    taf = ckpt.get("test_accuracies_full_list", [])
    tra = ckpt.get("train_accuracies", [])
    traf = ckpt.get("train_accuracies_full_list", [])
    tl = ckpt.get("test_losses", [])
    trl = ckpt.get("train_losses", [])
    iters = ckpt.get("iters", [])

    return {
        "name": os.path.basename(log_dir),
        "log_dir": log_dir,
        "msh_mode": config.get("msh_mode", "flat"),
        "msh_levels": config.get("msh_levels", ""),
        "sort_loss_mode": config.get("sort_loss_mode", "ctc"),
        "bp_steps": config.get("bp_steps", "0"),
        "msh_gate_init": config.get("msh_gate_init", ""),
        "msh_gate_sparsity": config.get("msh_gate_sparsity", "0.0"),
        "best_test_acc": max(ta) if ta else 0,
        "final_test_acc": ta[-1] if ta else 0,
        "best_test_acc_full": max(taf) if taf else 0,
        "final_test_acc_full": taf[-1] if taf else 0,
        "best_train_acc": max(tra) if tra else 0,
        "final_train_acc": tra[-1] if tra else 0,
        "best_train_acc_full": max(traf) if traf else 0,
        "final_train_acc_full": traf[-1] if traf else 0,
        "final_test_loss": tl[-1] if tl else 0,
        "final_train_loss": trl[-1] if trl else 0,
        "num_track_points": len(ta),
        "last_iter": iters[-1] if iters else 0,
        "train_acc_curve": str([round(x, 4) for x in tra]) if tra else "",
        "test_acc_curve": str([round(x, 4) for x in ta]) if ta else "",
        "test_loss_curve": str([round(x, 4) for x in tl]) if tl else "",
    }


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "csv_data/msh_results.csv"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    results = []
    for log_dir in LOG_DIRS:
        if not os.path.exists(log_dir):
            continue
        for sub in sorted(glob.glob(os.path.join(log_dir, "*"))):
            if os.path.isdir(sub):
                r = extract_experiment(sub)
                if r and "error" not in r:
                    results.append(r)

    if not results:
        print("No results found.")
        return

    fields = list(results[0].keys())
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nExported {len(results)} experiments to {output}\n")
    print(f"{'experiment':<30} {'msh_mode':<12} {'levels':<10} {'best_acc':>8} {'final_acc':>10} {'loss':>8}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: -x["best_test_acc"]):
        print(f"{r['name']:<30} {r['msh_mode']:<12} {r['msh_levels']:<10} "
              f"{r['best_test_acc']:>8.4f} {r['final_test_acc']:>10.4f} {r['final_test_loss']:>8.4f}")


if __name__ == "__main__":
    main()
