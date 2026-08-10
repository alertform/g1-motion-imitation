#!/usr/bin/env python3
"""多段策略的**逐段**评估。

为什么需要单独一个脚本：多段训练最典型的失败模式是「平均看着还行，
其实只学会了最容易的那几段」。整体均值会掩盖这种情况，必须逐段出数。

用法:
    python rl_eval_multi.py --ckpt runs/multi/policy.pkl --limit 8
    python rl_eval_multi.py --clips walk1_subject1,sprint1_subject2
"""
import argparse
import json
import pathlib

import numpy as np
import jax

import jax_compat  # noqa: F401
import rl_env
from rl_eval_mjx import eval_starts, rollout


def pick_clips(grade, names, limit):
    """与 rl_train.pick_clips 保持一致的选段逻辑。"""
    mf = pathlib.Path.home()/"tools"/"g1_dataset"/"manifest.json"
    rows = json.loads(mf.read_text())["motions"]
    if names:
        want = [n.strip() for n in names.split(",")]
        rows = [r for r in rows if any(w in r["name"] for w in want)]
    else:
        rows = [r for r in rows if r["grade"].startswith(grade)]
    rows.sort(key=lambda r: -r["frames"])
    return [r["name"] for r in (rows[:limit] if limit else rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/multi/policy_latest.pkl")
    ap.add_argument("--clips", default="", help="逗号分隔的动作名")
    ap.add_argument("--grade", default="可用")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=6, help="每段几个起点")
    ap.add_argument("--max-steps", type=int, default=1000)
    a = ap.parse_args()

    names = pick_clips(a.grade, a.clips, a.limit)
    ref, refv, refb, refc, refl = rl_env.load_reference(names)
    env = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=a.max_steps)
    lens = np.asarray(refl)

    import rl_play
    ck = pathlib.Path(a.ckpt)
    if not ck.is_absolute():
        ck = pathlib.Path.home()/"tools"/"rl"/ck
    if not ck.exists():
        raise SystemExit(f"存档不存在: {ck}")
    pol = rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU)

    print(f"存档 {ck.name}   {len(names)} 段 × {a.episodes} 起点 × "
          f"{a.max_steps} 步   引擎 MJX")
    print()
    print(f"  {'动作':<30}{'帧数':>7}{'前馈':>7}{'策略':>7}{'倍数':>7}"
          f"{'跑满':>7}{'关节°':>7}{'漂移cm':>8}")
    print("  " + "-"*80)

    rows = []
    for ci, nm in enumerate(names):
        starts = eval_starts(int(lens[ci]), a.episodes)
        s0, *_ = rollout(env, starts, None, a.max_steps, ci)
        s1, j1, r1, v1, _ = rollout(env, starts, pol, a.max_steps, ci)
        full = int((s1 >= a.max_steps).sum())
        gain = s1.mean()/max(s0.mean(), 1e-9)
        rows.append(dict(name=nm, n=int(lens[ci]), ff=s0.mean(),
                         pol=s1.mean(), gain=gain, full=full,
                         jerr=np.nanmedian(j1), drift=np.nanmedian(r1),
                         surv=s1.tolist()))
        print(f"  {nm:<30}{lens[ci]:>7}{s0.mean():>7.0f}{s1.mean():>7.0f}"
              f"{gain:>7.1f}{full:>4}/{a.episodes}{np.nanmedian(j1):>7.2f}"
              f"{np.nanmedian(r1):>8.2f}", flush=True)   # 逐段刷出，别等全跑完

    print()
    g = np.array([r["gain"] for r in rows])
    p = np.array([r["pol"] for r in rows])
    print(f"  策略/前馈  中位 {np.median(g):.1f}  范围 [{g.min():.1f}, {g.max():.1f}]")
    print(f"  存活均值   中位 {np.median(p):.0f}  范围 [{p.min():.0f}, {p.max():.0f}]")
    tot = sum(r["full"] for r in rows)
    print(f"  跑满比例   {tot}/{len(rows)*a.episodes} = "
          f"{tot/(len(rows)*a.episodes)*100:.0f}%")

    print()
    print("  === 最差的 3 段 ===")
    for r in sorted(rows, key=lambda r: r["pol"])[:3]:
        print(f"    {r['name']:<30} 存活 {r['pol']:>6.0f}  "
              f"倍数 {r['gain']:>5.1f}  各起点 {r['surv']}")

    spread = p.max()/max(p.min(), 1e-9)
    print()
    if spread > 3:
        print(f"  ⚠ 段间差距 {spread:.1f} 倍 —— 平均值掩盖了短板，"
              f"最差的段可能根本没学会")
    else:
        print(f"  段间差距 {spread:.1f} 倍，各段学习程度较均衡")


if __name__ == "__main__":
    main()
