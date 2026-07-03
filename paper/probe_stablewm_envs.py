#!/usr/bin/env python3
"""探 stable-worldmodel env 接口 (obs 结构 / success / goal / reward), 给 GymAdapter 设计用.

算力机先装包 (设代理, stable-wm 要拉 ogbench/dmc 数据):
    export http_proxy="http://public-proxy.qihoo.net:3128"
    export https_proxy="http://public-proxy.qihoo.net:3128"
    pip install 'stable-worldmodel[all]'

然后跑:
    python paper/probe_stablewm_envs.py

输出每个 env 的: obs 结构(image/state/dict), action_space, info keys (找 success/goal),
reward/term 结构. 据此设计 GymAdapter (把 gymnasium env 包成我们 dict obs 协议).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    import gymnasium as gym
    import stable_worldmodel as swm  # noqa: 触发 env 注册
    print("stable_worldmodel 注册成功, version=", getattr(swm, "__version__", "?"))
except Exception as e:
    print(f"装包失败或 import 错: {e}")
    print("先: pip install 'stable-worldmodel[all]'  (设代理)")
    sys.exit(1)

# 代表性 env: DMC locomotion(image/state) + Fetch(manipulation) + 经典 WM benchmark
ENVS = [
    "swm/PendulumDMControl-v0",
    "swm/ReacherDMControl-v0",
    "swm/CheetahDMControl-v0",
    "swm/CartpoleDMControl-v0",
    "swm/FetchReach-v3",
    "swm/FetchPush-v3",
    "swm/TwoRoom-v1",
    "swm/PushT-v1",
]


def probe(name, render_mode=None):
    try:
        env = gym.make(name, render_mode=render_mode) if render_mode else gym.make(name)
    except Exception as e:
        print(f"  [make {render_mode}] 失败: {e}")
        return
    try:
        obs, info = env.reset(seed=0)
        print(f"  obs_space: {env.observation_space}")
        print(f"  action_space: {env.action_space}  (shape {env.action_space.shape})")
        if isinstance(obs, dict):
            print(f"  obs(dict): keys={list(obs.keys())}")
            for k, v in obs.items():
                v = np.asarray(v)
                print(f"    {k}: shape={v.shape} dtype={v.dtype} range=[{v.min():.3f},{v.max():.3f}]")
        else:
            obs = np.asarray(obs)
            print(f"  obs: shape={obs.shape} dtype={obs.dtype} range=[{obs.min():.3f},{obs.max():.3f}]"
                  f"  -> {'IMAGE' if obs.ndim >= 3 else 'STATE'}")
        print(f"  info@reset keys: {list(info.keys())}")
        # 走几步, 看 reward/term/info 变化
        a = env.action_space.sample()
        for t in range(3):
            obs2, r, term, trunc, info2 = env.step(np.clip(a, env.action_space.low, env.action_space.high))
        print(f"  step: reward={r:.4f}  term={term}  trunc={trunc}")
        print(f"  info@step keys: {list(info2.keys())}")
        # 找 success / goal 信号
        for key in ("success", "is_success", "goal", "desired_goal", "achieved_goal",
                    "target", "distance", "success_rate"):
            if key in info2:
                v = info2[key]
                v = np.asarray(v)
                print(f"    >>> info[{key}]: {v.shape if v.shape else v}")
        env.close()
    except Exception as e:
        import traceback
        print(f"  probe 出错: {e}")
        traceback.print_exc()


for name in ENVS:
    print(f"\n=== {name} ===")
    print("[默认 render_mode]")
    probe(name, None)
    print("[render_mode=rgb_array]  (拿 image obs)")
    probe(name, "rgb_array")

print("\n探完. 把输出贴回, 我据此写 GymAdapter (obs→{state/pixels,goal}, success/goal 提取, partial 遮挡).")
