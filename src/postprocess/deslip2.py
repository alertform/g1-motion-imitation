#!/usr/bin/env python3
"""足滑清理 v2：同时处理漂移和抖动。

解剖数据显示 ground1 的脚滑由两部分构成，各占一半：
  - 净漂移 48%  接触段内脚整体移动了 4.8cm（中位）
  - 高频抖动 52% 路径长度 9.9cm 远大于净位移

两种成因要用不同手段：
  漂移 -> 根位置反向补偿（平移整个人，脚相对世界不动）
  抖动 -> 对接触段内的腿部关节角做低通（让脚的位置本身平稳）

只滤腿部关节，不动手臂和躯干——否则整体动作会糊。
"""
import argparse
import pathlib

import numpy as np
import mujoco
from scipy.signal import savgol_filter

MENAGERIE = pathlib.Path.home()/"mujoco-lab"/"mujoco_menagerie"/"unitree_g1"/"scene.xml"
FOOT_LINKS = {"left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"}
# qpos[7:] 里腿部关节的下标（前 12 个是双腿）
LEG_SLICE = slice(7, 19)


def segments(mask, min_len=5):
    out, s = [], None
    for i, on in enumerate(mask):
        if on and s is None:
            s = i
        elif not on and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask)))
    return out


class FK:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
        self.d = mujoco.MjData(self.m)
        self.b = {k: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, v)
                  for k, v in FOOT_LINKS.items()}

    def feet_xy(self, qpos):
        out = {}
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        for k, b in self.b.items():
            out[k] = self.d.xpos[b][:2].copy()
        return out

    def track(self, qpos, key):
        b = self.b[key]
        P = np.empty((len(qpos), 2))
        for i in range(len(qpos)):
            self.d.qpos[:] = qpos[i]
            mujoco.mj_forward(self.m, self.d)
            P[i] = self.d.xpos[b][:2]
        return P


def smooth_legs_in_contact(qpos, masks, keys, fk, window, min_seg):
    """接触段内对腿部关节做低通，压掉抖动。

    只在接触段内滤，且两端各留 ramp 帧做混合，避免在段边界产生台阶。
    """
    out = qpos.copy()
    T = len(qpos)
    # 每帧是否处于任一脚的接触段
    in_contact = np.zeros(T, bool)
    for k in FOOT_LINKS:
        for s, e in segments(masks[keys.index(k)], min_seg):
            in_contact[s:e] = True

    legs = qpos[:, LEG_SLICE]
    w = max(5, window | 1)
    if T > w:
        smoothed = savgol_filter(legs, w, 2, axis=0)
        # 只在接触帧应用，非接触帧保留原值；用软掩码过渡
        alpha = in_contact.astype(float)
        alpha = savgol_filter(alpha, w, 2) if T > w else alpha
        alpha = np.clip(alpha, 0, 1)[:, None]
        out[:, LEG_SLICE] = legs * (1 - alpha) + smoothed * alpha
    return out


def compensate_root(qpos, masks, keys, fk, min_seg, smooth):
    """按接触段锚点反向补偿根的水平位置，消除净漂移。"""
    T = len(qpos)
    P = {k: fk.track(qpos, k) for k in FOOT_LINKS}
    desired = np.zeros((T, 2))
    count = np.zeros(T)
    for k in FOOT_LINKS:
        for s, e in segments(masks[keys.index(k)], min_seg):
            anchor = np.median(P[k][s:e], axis=0)
            desired[s:e] += anchor - P[k][s:e]
            count[s:e] += 1
    hit = count > 0
    if not hit.any():
        return qpos.copy()
    desired[hit] /= count[hit, None]
    idx = np.flatnonzero(hit)
    filled = np.stack([np.interp(np.arange(T), idx, desired[idx, j]) for j in (0, 1)], 1)
    w = max(5, smooth | 1)
    if T > w:
        filled = savgol_filter(filled, w, 2, axis=0)
    out = qpos.copy()
    out[:, :2] += filled
    return out


def measure(qpos, masks, keys, fk, min_seg):
    res = {}
    for k in FOOT_LINKS:
        P = fk.track(qpos, k)
        slips, nets = [], []
        for s, e in segments(masks[keys.index(k)], min_seg):
            seg = P[s:e]
            if len(seg) > 1:
                slips.append(float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))))
                nets.append(float(np.linalg.norm(seg[-1] - seg[0])))
        res[k] = (float(np.median(slips)) if slips else np.nan,
                  float(np.median(nets)) if nets else np.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-seg", type=int, default=6)
    ap.add_argument("--leg-window", type=int, default=11)
    ap.add_argument("--root-smooth", type=int, default=9)
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()

    z = np.load(a.inp, allow_pickle=True)
    q = z["qpos"]
    masks, keys = z["contacts"], [str(x) for x in z["contact_keys"]]
    fk = FK()

    b = measure(q, masks, keys, fk, a.min_seg)
    print(f"  {'足':<12}{'路径':>9}{'净位移':>9}   ->{'路径':>9}{'净位移':>9}{'改善':>9}")
    print("  " + "-"*60)

    cur = q
    for _ in range(a.iters):
        cur = smooth_legs_in_contact(cur, masks, keys, fk, a.leg_window, a.min_seg)
        cur = compensate_root(cur, masks, keys, fk, a.min_seg, a.root_smooth)

    af = measure(cur, masks, keys, fk, a.min_seg)
    for k in FOOT_LINKS:
        d = 100*(af[k][0]/b[k][0]-1) if b[k][0] > 0 else 0
        print(f"  {k:<12}{b[k][0]*100:>8.1f}cm{b[k][1]*100:>8.1f}cm   ->"
              f"{af[k][0]*100:>8.1f}cm{af[k][1]*100:>8.1f}cm{d:>8.0f}%")

    out = pathlib.Path(a.out)
    np.savez_compressed(out, qpos=cur, fps=z["fps"], contacts=masks,
                        contact_keys=z["contact_keys"], ground=z["ground"])
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
