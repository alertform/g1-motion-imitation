#!/usr/bin/env python3
"""修复根位置的孤立瞬移帧，而不是整段剔除。

首次审计把 35 段判为「瞬移」而剔除，但查下来两类被混在一起：
    sprint1_subject4  6.19 m/s, 加速度 52.6 m/s²   -> 正常冲刺
    run2_subject1    21.41 m/s, 加速度 698  m/s²   -> 真瞬移
    walk3_subject4   27.57 m/s (走路 100km/h)      -> 真瞬移

原判据 15cm/帧 = 4.5 m/s，比人类快跑还慢，会把所有跑步误判。
**加速度**才是区分「高速」和「不连续」的判据。

而且真瞬移只占 0.1% 的帧——为 0.1% 丢掉整段是浪费。这里改成：
检出异常帧 -> 用两侧的合法帧线性插值 -> 轻平滑。
"""
import argparse
import pathlib

import numpy as np
from scipy.signal import savgol_filter

FPS = 30.0
# 人类速度上限参考：百米世界纪录约 12 m/s。超过即物理不可能。
MAX_SPEED = 12.0
# 人体运动加速度峰值一般 < 50 m/s²，取 4 倍余量。
MAX_ACC = 200.0


def detect_bad(q):
    """返回需要修复的帧掩码。"""
    v = np.append(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)*FPS, 0.0)
    a = np.concatenate([[0.0],
                        np.linalg.norm(np.diff(q[:, :3], n=2, axis=0), axis=1)*FPS**2,
                        [0.0]])
    bad = (v > MAX_SPEED) | (a > MAX_ACC)
    # 把异常帧两侧各一帧也标上——不连续通常牵连相邻帧
    out = bad.copy()
    idx = np.flatnonzero(bad)
    for i in idx:
        out[max(0, i-1):min(len(bad), i+2)] = True
    return out


def repair(q, bad, smooth=7):
    """线性插值填补异常帧的根位置，再轻平滑。关节角不动。"""
    out = q.copy()
    good = ~bad
    if good.sum() < 10 or bad.sum() == 0:
        return out, 0
    t = np.arange(len(q))
    for c in range(3):
        out[bad, c] = np.interp(t[bad], t[good], q[good, c])
    w = max(5, smooth | 1)
    if len(q) > w:
        out[:, :3] = savgol_filter(out[:, :3], w, 2, axis=0)
    return out, int(bad.sum())


def stats(q):
    v = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)*FPS
    a = np.linalg.norm(np.diff(q[:, :3], n=2, axis=0), axis=1)*FPS**2
    return v.max(), a.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(pathlib.Path.home()/"tools"/"g1_dataset"/"final"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(pathlib.Path(a.dir).glob("*.npz"))
    print(f"检查 {len(files)} 段   速度上限 {MAX_SPEED} m/s   加速度上限 {MAX_ACC} m/s²\n")
    print(f"  {'动作':<30}{'异常帧':>7}{'占比':>7}{'速度前→后':>17}{'加速度前→后':>19}")
    print("  " + "-"*82)
    n_fixed = n_clean = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        q = np.asarray(z["qpos"], dtype=np.float64)
        bad = detect_bad(q)
        if bad.sum() == 0:
            n_clean += 1
            continue
        v0, a0 = stats(q)
        q2, n = repair(q, bad)
        v1, a1 = stats(q2)
        print(f"  {f.stem:<30}{n:>7}{100*n/len(q):>6.1f}%"
              f"{v0:>8.1f}→{v1:<7.1f}{a0:>9.1f}→{a1:<9.1f}")
        if not a.dry_run:
            np.savez_compressed(f, qpos=q2, fps=z["fps"], contacts=z["contacts"],
                                contact_keys=z["contact_keys"], ground=z["ground"])
        n_fixed += 1
    print()
    print(f"  修复 {n_fixed} 段，本来就干净 {n_clean} 段"
          + ("   [dry-run，未写盘]" if a.dry_run else ""))


if __name__ == "__main__":
    main()
