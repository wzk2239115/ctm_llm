#!/usr/bin/env python3
"""定位 parity+JEPA 压制主任务的根因, 并验证自适应权重方案 (A/B/C) 是否解除压制。

背景: st04 里 parity+JEPA (w=0.1) 全部 final_iter=0 (acc 卡 0.499=随机)。
诊断发现不是 pool/crash 问题, 而是 w=0.1 的 JEPA 辅助 loss 梯度压制了
主任务 —— synch 表示被推向"可预测下一 tick"而非"对 parity 有判别力"。
本脚本扫 baseline / fixed(w,压制基线) / balance(A) / gate(B) / uncertainty(C),
看哪个自适应模式让 acc 脱离 0.5 (解除压制)。

算力机一键跑:
    cd /home/jovyan/h800fast/wangzekai/ctm_llm
    python paper/diagnose_parity_jepa_compute.py --steps 5000 --timeout 2400

本地 smoke (CPU, 小配置, 只看不崩):
    python paper/diagnose_parity_jepa_compute.py --local --steps 100 --timeout 600

策略: subprocess 直接调 baseline.tasks.parity.train (复用其成熟 setup),
解析 stdout 的 loss/acc 曲线 + stderr 的 traceback, 自动判定失败类型。

verdict 修过一个 bug: 旧版用 `acc<0.52 AND loss>0.69` 判 STALLED, 但 JEPA
辅助 loss 会拉低总 loss (制造"在学"假象), 导致 w=0.1 被误判 OK。现在纯看
acc (主任务真实信号): acc 卡随机 = STALLED, 与 loss 无关。
"""
import argparse, re, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = "/tmp/diag_parity_compute"

ALL_MODES = ["fixed", "balance", "gate", "uncertainty"]


def build_cmd(args, jepa_w, mode='fixed'):
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
        "--track_every", str(args.steps + 10),   # no mid-run gif/plot
        "--save_every", "999999",
        "--log_dir", LOG, "--device", str(args.device),
        "--lr", str(args.lr),
        "--gradient_clipping", str(args.grad_clip),
    ]
    if jepa_w > 0:
        cmd += ["--cross_tick_jepa_weight", str(jepa_w),
                "--cross_tick_jepa_hidden_dim", "128",
                "--cross_tick_jepa_predictor_depth", "2",
                "--cross_tick_jepa_dropout", "0.0",
                "--jepa_weight_mode", mode]
        if mode == 'balance':
            cmd += ["--jepa_balance_ratio", str(args.balance_ratio)]
        elif mode == 'gate':
            cmd += ["--jepa_gate_threshold", str(args.gate_threshold),
                    "--jepa_gate_temp", str(args.gate_temp)]
        elif mode == 'uncertainty':
            cmd += ["--jepa_log_sigma_init", str(args.log_sigma_init)]
    return cmd


def classify(label, rc, out, accs, losses):
    """从 returncode + acc 曲线判定失败类型。

    STALLED 判据只看 acc (主任务真实信号), 不看 loss —— 因为 JEPA 辅助 loss
    会拉低总 loss 制造"在学"假象 (正是 w=0.1 的陷阱)。
    """
    lower = out.lower()
    if rc != 0:
        if "out of memory" in lower or "cuda oom" in lower:
            return ("OOM", "显存不够。建议: --batch 32 或 --d_model 512。")
        if re.search(r"loss=nan|nan\b", out):
            return ("NAN", "loss 变 NaN(数值不稳)。建议: --grad_clip 0.5 / 降 lr / 开 --use_amp。")
        tb = [l for l in out.splitlines() if "Error" in l or "error" in l]
        last = tb[-1].strip()[:90] if tb else "(见 stderr)"
        return ("CRASH", f"异常退出: {last}")
    if not accs:
        return ("CRASH", "没产出 Accuracy 数据(可能 setup 阶段崩, 见 stderr)。")
    if any(a != a for a in accs):  # nan in acc
        return ("NAN", "Accuracy 出现 NaN。")
    final_acc = accs[-1]
    last_chunk = accs[max(0, len(accs) // 5):]
    mean_last = sum(last_chunk) / len(last_chunk) if last_chunk else 0
    # 纯 acc 判据 (修过 bug): acc 卡随机 = 主任务没学, 与 loss 无关
    if mean_last < 0.52:
        return ("STALLED", f"acc 卡在 {mean_last:.3f}(随机), 主任务没学。"
                f"注意: 总 loss 可能因辅助项下降, 但 acc 是主任务真实信号。")
    if mean_last < 0.55:
        return ("WEAK", f"acc 勉强脱离随机 ({mean_last:.3f}), 但学得很慢。")
    return ("OK", f"在学: final acc={final_acc:.3f}, 最后段均值 {mean_last:.3f}。")


def run(label, args, jepa_w, mode='fixed'):
    shutil.rmtree(LOG, ignore_errors=True)
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    cmd = build_cmd(args, jepa_w, mode)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(ROOT), timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f">>> TIMEOUT (>{args.timeout}s) — 训练卡住(可能死锁/极慢)。")
        return "TIMEOUT", [], []
    out = r.stdout + r.stderr
    accs = [float(x.rstrip(".")) for x in re.findall(r"Accuracy=([0-9.]+)", out)]
    losses = [float(x.rstrip(".")) for x in re.findall(r"Loss=([0-9.]+)", out)]
    print(f"returncode={r.returncode}  acc_points={len(accs)}  loss_points={len(losses)}")
    if accs:
        n = len(accs)
        for i in sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])):
            if i < n:
                li = losses[i] if i < len(losses) else float('nan')
                print(f"  step~{i:5d}  acc={accs[i]:.3f}  loss={li:.3f}")
    verdict, msg = classify(label, r.returncode, out, accs, losses)
    print(f">>> VERDICT: {verdict} — {msg}")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-25:])
        print("--- stderr/traceback 尾部 ---")
        print(tail)
    return verdict, accs, losses


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
    ap.add_argument("--steps", type=int, default=40, help="训练步数")
    ap.add_argument("--timeout", type=int, default=600, help="单配置超时秒数")
    ap.add_argument("--device", type=int, default=0)
    # JEPA mode 扫描
    ap.add_argument("--jepa_w_test", type=float, default=0.1,
                    help="测试用的 JEPA base weight (默认 0.1 = 已知压制 parity 的配置)")
    ap.add_argument("--jepa_mode", type=str, default="fixed,balance,gate,uncertainty",
                    help="逗号分隔的 mode 列表; fixed=压制基线, 其余=待验证的自适应方案")
    # 自适应方案子参数
    ap.add_argument("--balance_ratio", type=float, default=0.3, help="[A balance] JEPA/main ratio")
    ap.add_argument("--gate_threshold", type=float, default=0.6, help="[B gate] acc EMA 半开点")
    ap.add_argument("--gate_temp", type=float, default=0.05, help="[B gate] sigmoid 温度")
    ap.add_argument("--log_sigma_init", type=float, default=0.0, help="[C uncertainty] log sigma 初始")
    # 本地 smoke
    ap.add_argument("--local", action="store_true", help="CPU 小配置 smoke (只看不崩)")
    args = ap.parse_args()
    import math
    assert int(math.sqrt(args.seq))**2 == args.seq, "seq 必须是完全平方数"

    if args.local:
        args.d_model, args.d_input = 128, 64
        args.seq, args.batch, args.iterations = 16, 32, 10
        args.memory_length = 5
        args.device = -1
        if args.steps > 200:
            args.steps = 200

    modes = [m.strip() for m in args.jepa_mode.split(",") if m.strip()]
    dev_label = "CPU" if args.device < 0 else ("GPU: " + __import__('torch').cuda.get_device_name(0)) \
        if args.device >= 0 else "CPU"
    print(f"parity 诊断 | d_model={args.d_model} seq={args.seq} "
          f"batch={args.batch} ticks={args.iterations} steps={args.steps}")
    print(f"设备: {dev_label}")
    print(f"JEPA base weight (测试配置): {args.jepa_w_test}")
    print(f"扫描 modes: {modes}")
    print(f"  [A balance] ratio={args.balance_ratio}  [B gate] thresh={args.gate_threshold} temp={args.gate_temp}"
          f"  [C uncertainty] log_sigma0={args.log_sigma_init}")

    results = {}
    # 0) baseline 控制组
    v, _, _ = run("0) BASELINE (jepa=0) — 控制组", args, 0.0, 'fixed')
    results["baseline"] = v
    # 各 mode (都用同一个 base weight, 区分 adaptive 方案)
    for i, m in enumerate(modes, 1):
        tag = f"{m}(w={args.jepa_w_test})"
        v, _, _ = run(f"{i}) {m.upper()}  base_w={args.jepa_w_test}", args, args.jepa_w_test, m)
        results[m] = v

    print("\n" + "=" * 70)
    print("FINAL VERDICT — 是否解除 JEPA 压制")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:14s} -> {v}")

    b = results["baseline"]
    if b not in ("OK", "WEAK"):
        print(f"\nbaseline={b} -> 配置本身不稳(步数太少或崩), 加 --steps 再判。")
        return

    print(f"\nbaseline 在学 ({b})。判各 mode 是否让 acc 脱离随机(解除压制):")
    fixed = results.get("fixed")
    if fixed:
        print(f"  fixed    : {fixed}  {'<- 压制复现 (acc 卡 0.5)' if fixed == 'STALLED' else '(意外没压制?)'}")
    for k, v in results.items():
        if k in ("baseline", "fixed"):
            continue
        if v in ("OK", "WEAK"):
            print(f"  {k:9s}: {v}  ✓ 解除压制 — acc 脱离随机, 主任务在学")
        elif v == "STALLED":
            print(f"  {k:9s}: {v}  ✗ 仍压制 — 该方案没救回主任务")
        else:
            print(f"  {k:9s}: {v}  ? 非压制问题 (崩/NaN/OOM), 见上")
    winners = [k for k, v in results.items()
               if k not in ("baseline", "fixed") and v in ("OK", "WEAK")]
    if winners:
        print(f"\n结论: 自适应方案 {winners} 解除了 w={args.jepa_w_test} 的压制。")
        print("下一步: 算力机长跑 (--steps 30000+) 看 final acc 能否追上 baseline。")
    else:
        print(f"\n结论: 没有方案解除压制。调子参数 (--balance_ratio / --gate_threshold) 再试,")
        print("      或确认步数够 (压制下 acc 本就慢, 至少 3000 步才能区分 STALLED vs WEAK)。")


if __name__ == "__main__":
    main()
