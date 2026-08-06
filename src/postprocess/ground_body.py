#!/usr/bin/env python3
"""全身对地：以整个身体的最低碰撞点为基准，而不是只看脚。

为什么必须这样：实测 ground1 里只有 36.7% 的帧脚是全身最低点。趴着时
髋部（hip_roll 24.2% + hip_yaw 23%）才是贴地的部位。只按脚对地会让
44% 的帧躯干穿进地面，最深 8.7cm——视觉上就是"陷进地里再弹出来"。

做法：逐帧扫描所有参与碰撞的 geom（mesh 按顶点算），取最低点，
把根位置竖直平移使其落到 0（允许一点点穿透，跟真实接触一致）。

弹跳来自最低点在不同部位之间切换时的跳变，用两级平滑压掉：
  - 先对原始修正量做中值滤波（去掉单帧离群）
  - 再做 Savitzky-Golay（保形低通）

只改 qpos[2]，关节角完全不动。
"""
import argparse
import pathlib

import numpy as np
import mujoco
from scipy.signal import savgol_filter, medfilt


class BodyProbe:
    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.geoms = [g for g in range(self.m.ngeom)
                      if g != floor and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        # 预取网格顶点，避免逐帧重复切片
        self.verts = {}
        for g in self.geoms:
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                self.verts[g] = np.asarray(self.m.mesh_vert[a:a+n], dtype=np.float64)

    def lowest(self, qpos):
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        lo = np.inf
        for g in self.geoms:
            if g in self.verts:
                R = d.geom_xmat[g].reshape(3, 3)
                z = float((self.verts[g] @ R.T)[:, 2].min() + d.geom_xpos[g][2])
            else:
                z = float(d.geom_xpos[g][2]) - float(m.geom_size[g][0])
            if z < lo:
                lo = z
        return lo

    def lowest_named(self, qpos):
        """返回 (最低高度, 所属 body 名)，用于诊断。"""
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        lo, who = np.inf, -1
        for g in self.geoms:
            if g in self.verts:
                R = d.geom_xmat[g].reshape(3, 3)
                z = float((self.verts[g] @ R.T)[:, 2].min() + d.geom_xpos[g][2])
            else:
                z = float(d.geom_xpos[g][2]) - float(m.geom_size[g][0])
            if z < lo:
                lo, who = z, g
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[who]) or "?"
        return lo, name


def ground_body(qpos, probe, med=5, sg=11, allow_pen=0.005):
    """把全身最低点压到 -allow_pen（留一点穿透，接触才自然）。"""
    T = len(qpos)
    lows = np.array([probe.lowest(qpos[i]) for i in range(T)])
    shift = -(lows + allow_pen)

    if med >= 3:
        shift = medfilt(shift, med | 1)        # 先去单帧离群
    w = max(5, sg | 1)
    if T > w:
        shift = savgol_filter(shift, w, 2)     # 再保形低通

    out = qpos.copy()
    out[:, 2] += shift
    return out, shift


def report(qpos, probe, label, step=5):
    lows = np.array([probe.lowest(qpos[i]) for i in range(0, len(qpos), step)])
    pen1 = 100 * np.mean(lows < -0.01)
    pen5 = 100 * np.mean(lows < -0.05)
    air2 = 100 * np.mean(lows > 0.02)
    print(f"  {label:<10} 最低点中位 {np.median(lows):+.4f}   "
          f"穿地>1cm {pen1:5.1f}%   >5cm {pen5:4.1f}%   悬空>2cm {air2:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default=str(pathlib.Path.home()/"mujoco-lab"/
                                         "mujoco_menagerie"/"unitree_g1"/"scene.xml"))
    ap.add_argument("--med", type=int, default=5)
    ap.add_argument("--sg", type=int, default=11)
    ap.add_argument("--allow-pen", type=float, default=0.005)
    a = ap.parse_args()

    z = np.load(a.inp, allow_pickle=True)
    q = z["qpos"]
    probe = BodyProbe(a.xml)

    report(q, probe, "修正前")
    fixed, shift = ground_body(q, probe, a.med, a.sg, a.allow_pen)
    report(fixed, probe, "修正后")

    dz = np.abs(np.diff(fixed[:, 2]))
    print(f"  平均竖直修正 {np.mean(np.abs(shift))*100:.2f} cm   "
          f"根高度单帧跳变>2cm: {int(np.sum(dz > 0.02))} 次（最大 {dz.max()*100:.1f}cm）")

    out = pathlib.Path(a.out)
    np.savez_compressed(out, qpos=fixed, fps=z["fps"], contacts=z["contacts"],
                        contact_keys=z["contact_keys"], ground=z["ground"])
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
