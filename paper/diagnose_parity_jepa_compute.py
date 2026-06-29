#!/usr/bin/env python3
"""定位 parity+JEPA 在算力机上 final_iter=0 的精确原因。

算力机上一键跑:
    cd /home/jovyan/h800fast/wangzekai/ctm_llm
    python paper/diagnose_parity_jepa_compute.py

策略: subprocess 直接调 baseline.tasks.parity.train(复用其成熟 setup),
       用算力机真实配置(d_model=1024, seq=64, batch=64, iterations=75),
       跑足够步数, 解析 stdout 的 loss/acc 曲线 + stderr 的 traceback,
       自动判定失败类型: OOM / NAN / CRASH / STALLED / COLLAPSE / OK。

baseline 和 JEPA 各跑一遍对比, 末尾给 VERDICT + 克服建议。

(不用手动构造 model — parity backbone 是 lazy + prediction_reshaper 特殊,
 subprocess 调 train.py 最稳, 也最接近算力机真实训练路径。)
"""
import argparse, re, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = "/tmp/diag_parity_compute"


def build_cmd(args, jepa_w):
    cmd = [
        sys.executable, "-m", "baseline.tasks.parity.train",
        "--model_type", "ctm",
        "--iterations", str(args.iterations),
        "--memory_length", str(args.memory_length),
        "--parity_sequence_length", str(args.seq),
        "--d_model", str(args.d_model), "--d_input", str(args.d_input),
        "--n_synch_out", "32", "--n_synch_action", "32",
        "--memory_hidden_dims", "16",
        "--synapse_depth", "1", "--heads", str(args.heads),
        "--backbone_type", "parity_backbone",
        "--positional_embedding_type", "custom-rotational-1d",
        "--neuron_select_type", "random",
        "--batch_size", str(args.batch), "--batch_size_test", str(args.batch),
        "--training_iterations", str(args.steps),
        "--track_every", str(args.steps + 10),   # no mid-run gif/plot (慢/可能崩)
        "--save_every", "999999",
        "--log_dir", LOG, "--device", "0",
        "--lr", str(args.lr),
        "--gradient_clipping", str(args.grad_clip),
    ]
    if jepa_w > 0:
        cmd += ["--cross_tick_jepa_weight", str(jepa_w),
                "--cross_tick_jepa_hidden_dim", "128",
                "--cross_tick_jepa_predictor_depth", "2",
                "--cross_tick_jepa_dropout", "0.0"]
    return cmd


def classify(label, rc, out, accs, losses):
    """从 returncode + stderr 关键词 + acc 曲线判定失败类型。"""
    lower = out.lower()
    if rc != 0:
        if "out of memory" in lower or "cuda oom" in lower:
            return ("OOM", "显存不够。建议: --batch 32 或 --d_model 512。")
        if re.search(r"loss=nan|nan\b", out):
            return ("NAN", "loss 变 NaN(数值不稳)。建议: --grad_clip 0.5 / "
                           "改 stablemax_ce / 降 lr 5e-5 / 开 --use_amp。")
        # extract last traceback line
        tb = [l for l in out.splitlines() if "Error" in l or "error" in l]
        last = tb[-1].strip()[:90] if tb else "(见 stderr)"
        return ("CRASH", f"异常退出: {last}")
    if not accs:
        return ("CRASH", "没产出 Accuracy 数据(可能 setup 阶段崩, 见 stderr)。")
    # 跑完了, 看曲线
    import math
    if any(a != a for a in accs):  # nan in acc
        return ("NAN", "Accuracy 出现 NaN。")
    final_acc = accs[-1]
    last_chunk = accs[max(0, len(accs) // 5):]
    mean_last = sum(last_chunk) / len(last_chunk) if last_chunk else 0
    if mean_last < 0.52 and (losses and losses[-1] > 0.69):
        return ("STALLED", f"acc 卡在 {mean_last:.3f}(随机), loss 不降。"
                f"可能步数太少({len(accs)}步)或 idea 干扰, 但非崩溃。")
    return ("OK", f"在学: final acc={final_acc:.3f}, 最后段均值 {mean_last:.3f}。")


def run(label, args, jepa_w):
    shutil.rmtree(LOG, ignore_errors=True)
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    cmd = build_cmd(args, jepa_w)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(ROOT), timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f">>> TIMEOUT (>{args.timeout}s) — 训练卡住(可能死锁/极慢)。")
        return "TIMEOUT"
    out = r.stdout + r.stderr
    accs = [float(x.rstrip(".")) for x in re.findall(r"Accuracy=([0-9.]+)", out)]
    losses = [float(x.rstrip(".")) for x in re.findall(r"Loss=([0-9.]+)", out)]
    # 只保留训练步的(去掉初始 eval 的 Where_certain=0 那批)
    print(f"returncode={r.returncode}  acc_points={len(accs)}  loss_points={len(losses)}")
    # 打印轨迹采样
    if accs:
        n = len(accs)
        for i in sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])):
            if i < n:
                print(f"  step~{i:4d}  acc={accs[i]:.3f}  loss={losses[i] if i < len(losses) else float('nan'):.3f}")
    verdict, msg = classify(label, r.returncode, out, accs, losses)
    print(f">>> VERDICT: {verdict} — {msg}")
    # 崩溃时打印 traceback 尾部
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-25:])
        print("--- stderr/traceback 尾部 ---")
        print(tail)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # parity paper 真实配置(算力机 st00/st04)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--d_input", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=75, help="n_ticks")
    ap.add_argument("--memory_length", type=int, default=25)
    ap.add_argument("--seq", type=int, default=64, help="parity_sequence_length (perfect square)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=40, help="训练步数(40步够看崩/NaN)")
    ap.add_argument("--timeout", type=int, default=600, help="单配置超时秒数")
    args = ap.parse_args()
    import math
    assert int(math.sqrt(args.seq))**2 == args.seq, "seq 必须是完全平方数"

    print(f"parity 诊断 (算力机真实配置) | d_model={args.d_model} seq={args.seq} "
          f"batch={args.batch} ticks={args.iterations} steps={args.steps}")
    print(f"GPU: {__import__('torch').cuda.get_device_name(0)}")

    results = {}
    results["baseline"] = run("1) BASELINE (jepa=0) — 控制组", args, 0.0)
    results["jepa_w0.1"] = run("2) JEPA w=0.1 — 算力机 killed 的配置", args, 0.1)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:12s} -> {v}")

    b, j = results["baseline"], results["jepa_w0.1"]
    if b == "OK" and j in ("OOM", "NAN", "CRASH"):
        print(f"\n结论: baseline 正常但 JEPA 崩 -> JEPA 特有问题({j})。")
        if j == "OOM": print("  克服: JEPA predictor 占额外显存, 降 batch 到 48。")
        if j == "NAN": print("  克服: 加强 grad_clip(0.5) / stablemax_ce / 降 lr。")
    elif b in ("OOM", "NAN", "CRASH"):
        print(f"\n结论: baseline 也崩({b}) -> 不是 JEPA 问题, 是 parity 大配置本身不稳。")
        print("  克服: 降 batch/d_model, 或查算力机当时是否多任务抢显存。")
    elif b == "OK" and j == "OK":
        print("\n结论: 两者都不崩 -> 算力机 final_iter=0 是 pool/checkpoint 工程问题。")
        print("  下一步: cat runs/metrics/st04*parity*.fail.json + tail logs/ 对应 .log")
    elif b == "OK" and j == "STALLED":
        print("\n结论: JEPA 不崩但训练停滞 -> idea 接上了但干扰学习, 调小 jepa_weight。")
    else:
        print(f"\n结论: baseline={b}, jepa={j}。看上面各 VERDICT 行的详细建议。")


if __name__ == "__main__":
    main()
