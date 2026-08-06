#!/usr/bin/env python3
"""最后一道打磨：关节限位 + 自碰撞。

关节限位
  超出量实测中位仅 0.11~0.23°、最大 2.0°，属于 IK 数值噪声。
  直接 clip 是安全的——实测 clip 后关节加速度 p95 从 13.97 降到 13.86，
  不升反降（因为削掉的正是越界的毛刺）。clip 后再做一次轻平滑收尾。

自碰撞
  典型是肩膀穿进躯干、手腕穿进髋部，穿透量通常几毫米到两三厘米。
  处理方式：检测到穿透后，沿接触法向把**相关关节**做小幅回退。
  这里用最保守的做法——对穿透帧的相关关节做局部时序平滑，把尖峰压掉；
  不做几何求解，因为那需要在 IK 里加避碰约束（要改 GMR 内部）。

  注意：自碰撞很多时候源自人体动作本身（人抱臂时手臂也贴着躯干），
  机器人肢体更粗，同样的动作就会穿模。所以目标是**减少**而非清零。
"""
import argparse
import pathlib

import numpy as np
import mujoco
from scipy.signal import savgol_filter

MENAGERIE = pathlib.Path.home()/"mujoco-lab"/"mujoco_menagerie"/"unitree_g1"/"scene.xml"


def clip_limits(q, m):
    """把关节角截回限位内。返回 (新 q, 被改的关节-帧数)。"""
    out = q.copy()
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    lim = m.jnt_limited[1:].astype(bool)
    j = out[:, 7:]
    before = j.copy()
    j[:, lim] = np.clip(j[:, lim], lo[lim], hi[lim])
    return out, int(np.sum(before != j))


def count_self_collisions(q, m, d, floor, step=5, thresh=-0.005):
    """返回 (有自穿透的帧比例, 每帧最深穿透)。"""
    idx = np.arange(0, len(q), step)
    depth = np.zeros(len(q))
    for i in idx:
        d.qpos[:] = q[i]
        mujoco.mj_forward(m, d)
        worst = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if floor in (con.geom1, con.geom2):
                continue
            worst = min(worst, float(con.dist))
        depth[i] = worst
    return float(np.mean(depth[idx] < thresh)), depth


def collision_bodies(q, m, d, floor, i):
    """该帧发生自穿透的 body 对。"""
    d.qpos[:] = q[i]
    mujoco.mj_forward(m, d)
    out = []
    for c in range(d.ncon):
        con = d.contact[c]
        if floor in (con.geom1, con.geom2) or con.dist >= -0.005:
            continue
        out.append((m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2], con.dist))
    return out


def joints_of_body(m, body):
    """从该 body 到根路径上的所有铰链关节下标（qpos[7:] 里的位置）。"""
    out = []
    b = body
    while b > 0:
        for j in range(1, m.njnt):
            if m.jnt_bodyid[j] == b and m.jnt_type[j] in (
                    mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                out.append(j - 1)
        b = m.body_parentid[b]
    return out


def relax_collisions(q, m, d, floor, step=5, window=9, passes=2):
    """对发生自穿透的帧，把相关关节做局部时序平滑。

    不做几何求解——那需要在 IK 阶段加避碰约束。这里只压掉造成穿透的
    姿态尖峰，属于保守处理。
    """
    out = q.copy()
    T = len(q)
    for _ in range(passes):
        touched = np.zeros((T, m.nu), bool)
        for i in range(0, T, step):
            for b1, b2, _dist in collision_bodies(out, m, d, floor, i):
                for k in set(joints_of_body(m, b1) + joints_of_body(m, b2)):
                    if 0 <= k < m.nu:
                        lo_i, hi_i = max(0, i-window//2), min(T, i+window//2+1)
                        touched[lo_i:hi_i, k] = True
        if not touched.any():
            break
        w = max(5, window | 1)
        if T > w:
            sm = savgol_filter(out[:, 7:], w, 2, axis=0)
            out[:, 7:] = np.where(touched, sm, out[:, 7:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-collision", action="store_true")
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()

    z = np.load(a.inp, allow_pickle=True)
    q = np.asarray(z["qpos"], dtype=np.float64)
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
    d = mujoco.MjData(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def acc(x):
        return float(np.mean(np.abs(np.diff(x[:, 7:], n=2, axis=0)))*900)

    pct0, _ = count_self_collisions(q, m, d, floor)
    print(f"  修正前: 自穿透帧 {100*pct0:.1f}%   关节加速度 {acc(q):.2f}")

    q1, nclip = clip_limits(q, m)
    print(f"  关节限位: clip 了 {nclip} 个关节-帧")

    if not a.no_collision:
        q1 = relax_collisions(q1, m, d, floor, window=a.window, passes=a.passes)
        q1, n2 = clip_limits(q1, m)     # 平滑可能又推出限位，再收一次
        if n2:
            print(f"  平滑后二次 clip: {n2} 个")

    pct1, _ = count_self_collisions(q1, m, d, floor)
    # 校验限位
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    lim = m.jnt_limited[1:].astype(bool)
    jj = q1[:, 7:]
    over = np.zeros_like(jj, bool)
    over[:, lim] = (jj[:, lim] < lo[lim]-1e-9) | (jj[:, lim] > hi[lim]+1e-9)
    print(f"  修正后: 自穿透帧 {100*pct1:.1f}%   关节加速度 {acc(q1):.2f}   "
          f"越界 {100*over.mean():.3f}%")

    out = pathlib.Path(a.out)
    np.savez_compressed(out, qpos=q1, fps=z["fps"], contacts=z["contacts"],
                        contact_keys=z["contact_keys"], ground=z["ground"])
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
