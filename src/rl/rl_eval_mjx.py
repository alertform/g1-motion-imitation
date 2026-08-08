#!/usr/bin/env python3
"""在 MJX 里评估策略——和训练同一个物理引擎。

为什么需要它：MJX 与 CPU MuJoCo 的接触求解有实质差异。实测同一段
零动作前馈，5 个起点里 4 个吻合（差≤12 步），但有一个起点 CPU 跑出
371 步而 MJX 只有 53 步。用 CPU 评估 MJX 训练出的策略会得出错误结论。

CPU 版（rl_eval.py）仍然有用——它更接近真机，适合看 sim2real 差距。
但判断「训练有没有在起作用」必须用这个。

用法:
    python rl_eval_mjx.py --ckpt runs/walk/policy_latest.pkl
    python rl_eval_mjx.py --zero          # 只测零动作前馈基线
"""
import argparse
import pathlib

import numpy as np
import jax
import jax.numpy as jp
from brax.envs.base import State

import jax_compat  # noqa: F401
import rl_env

METRIC_KEYS = ("r_body", "r_pose", "r_orient", "r_root", "r_rvel", "r_jvel",
               "r_alive", "r_effort", "body_err", "pose_err", "root_err",
               "rvel_err")

# 起点预留：起点集**必须与 max_steps 无关**，否则不同步数上限的评估
# 之间无法纵向比较。踩过的坑：用 --max-steps 1000 和 1500 各评一次，
# linspace 的终点跟着变，16 个起点整体挪位，我却把两次结果当同一组
# 起点做了对比，得出「深蹲段已解决」的错误结论——实际那个起点
# 根本不在第二次的采样里。
EVAL_RESERVE = 1500     # 固定预留，与 max_steps 无关
LOOKAHEAD_PAD = 8       # 给 LOOKAHEAD 和 +1 索引留的余量


def eval_starts(T, episodes):
    """固定起点集，只由片段长度和起点数量决定。"""
    span = max(0, T - EVAL_RESERVE - LOOKAHEAD_PAD)
    return np.linspace(0, span, episodes).astype(int)


def make_state(env, clip, starts):
    """从指定参考帧构造初始 state，绕开 reset 的随机起点。"""
    n = len(starts)
    st_arr = jp.asarray(starts)

    def one(i):
        q, v = env._ref_at(clip, st_arr[i])
        return env.pipeline_init(q, v)

    data = jax.vmap(one)(jp.arange(n))
    zero_act = jp.zeros((n, rl_env.NU))
    obs = jax.vmap(lambda d, s, a: env._obs(d, clip, s, a))(data, st_arr, zero_act)
    return State(
        data, obs, jp.zeros(n), jp.zeros(n),
        {k: jp.zeros(n) for k in METRIC_KEYS},
        {"clip": jp.full((n,), clip, dtype=jp.int32),
         "start": st_arr, "step": st_arr, "last_act": zero_act,
         "rng": jax.random.split(jax.random.PRNGKey(0), n)})


def rollout(env, starts, policy, max_steps, clip=0):
    """跑到全部结束或步数上限，返回每个起点的存活步数与跟踪误差。"""
    st = make_state(env, clip, starts)
    step_fn = jax.jit(jax.vmap(env.step))
    n = len(starts)

    alive = np.ones(n, dtype=bool)
    surv = np.full(n, max_steps, dtype=int)
    pose_acc = [[] for _ in range(n)]
    root_acc = [[] for _ in range(n)]      # 根位置误差，速度奖励的直接目标
    rvel_acc = [[] for _ in range(n)]
    act_mag = []
    key = jax.random.PRNGKey(0)

    for i in range(max_steps):
        if policy is None:
            act = jp.zeros((n, rl_env.NU))
        else:
            act = jp.stack([policy(st.obs[k], key)[0] for k in range(n)])
        act_mag.append(float(jp.abs(act).mean()))
        st = step_fn(st, act)

        pe = np.asarray(st.metrics["pose_err"])
        re = np.asarray(st.metrics["root_err"])
        ve = np.asarray(st.metrics["rvel_err"])
        for k in range(n):
            if alive[k]:
                pose_acc[k].append(pe[k])
                root_acc[k].append(re[k])
                rvel_acc[k].append(ve[k])

        done = np.asarray(st.done).astype(bool)
        newly = alive & done
        surv[newly] = i + 1
        alive &= ~done
        if not alive.any():
            break

    # pose_err 是 rad² 均值 -> 度；root_err 是 m² -> cm；rvel_err 是 (m/s)² -> m/s
    jerr = np.array([np.degrees(np.sqrt(np.mean(p))) if p else np.nan
                     for p in pose_acc])
    rerr = np.array([np.sqrt(np.mean(p))*100 if p else np.nan
                     for p in root_acc])
    verr = np.array([np.sqrt(np.mean(p)) if p else np.nan
                     for p in rvel_acc])
    return surv, jerr, rerr, verr, float(np.mean(act_mag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/walk/policy_latest.pkl")
    ap.add_argument("--clip-name", default="walk1_subject1")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--zero", action="store_true", help="只测零动作基线")
    ap.add_argument("--starts", default="",
                    help="逗号分隔的起始帧，指定后忽略 --episodes（用于定点复查）")
    a = ap.parse_args()

    ref, refv, refb, refc = rl_env.load_reference([a.clip_name])
    env = rl_env.G1Imitate(ref, refv, refb, refc, ep_len=a.max_steps)

    if a.starts:
        starts = np.array([int(s) for s in a.starts.split(",")])
    else:
        starts = eval_starts(env._T, a.episodes)

    if a.max_steps > EVAL_RESERVE:
        print(f"  ⚠ max_steps={a.max_steps} 超过预留 {EVAL_RESERVE}，"
              f"末尾起点会跑过参考末端（参考帧钳制在最后一帧）")

    print(f"引擎 MJX（与训练一致）   动作 {a.clip_name}   "
          f"{len(starts)} 个起点 × {a.max_steps} 步")
    print(f"起始帧: {starts.tolist()}")
    print()

    rows = []
    print("  零动作前馈基线…")
    rows.append(("零动作前馈",) + rollout(env, starts, None, a.max_steps))

    if not a.zero:
        import rl_play
        ck = pathlib.Path(a.ckpt)
        if not ck.is_absolute():
            ck = pathlib.Path.home()/"tools"/"rl"/ck
        if not ck.exists():
            raise SystemExit(f"存档不存在: {ck}")
        pol = rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU)
        print(f"  训练策略 {ck.name}…")
        rows.append(("训练后策略",) + rollout(env, starts, pol, a.max_steps))

    print()
    print(f"  {'':>12} {'存活均值':>9} {'中位':>7} {'关节误差°':>10} "
          f"{'根漂移cm':>10} {'根速度误差':>11} {'|动作|':>8}")
    for lbl, s, j, r, v, m in rows:
        print(f"  {lbl:>12} {np.mean(s):>9.1f} {np.median(s):>7.1f} "
              f"{np.nanmedian(j):>10.2f} {np.nanmedian(r):>10.2f} "
              f"{np.nanmedian(v):>9.3f}m/s {m:>8.4f}")
    print()
    for lbl, s, j, r, v, m in rows:
        print(f"  {lbl:>12} 各起点存活 {s.tolist()}")

    if len(rows) == 2:
        gain = np.mean(rows[1][1]) / max(np.mean(rows[0][1]), 1e-9)
        print()
        print(f"  策略 / 前馈 = {gain:.2f}")
        print("  " + ("-> 训练在起作用" if gain > 1.05 else
                      "-> 策略未超过前馈，RL 尚未产生价值"))


if __name__ == "__main__":
    main()
