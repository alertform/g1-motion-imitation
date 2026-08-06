#!/usr/bin/env python3
"""批量审计 + 筛选，输出可用于 RL 训练的数据集清单。

不只是逐段报告，而是给出：
  1. 每段的量化指标（可排序、可设阈值）
  2. 按 RL 训练的实际需求分级（而不是笼统的"有几个问题"）
  3. 一份 manifest.json，训练脚本直接读

分级依据（见对话里的讨论）：
  致命   NaN / 关节越界 / 瞬移  -> 环境初始化就会炸，必须剔除
  重要   接触质量、对地一致性    -> 奖励函数直接用，差了会教坏策略
  次要   自碰撞、平滑度          -> RL 自己能容忍
  忽略   力矩可行性              -> 参考动作本就不需要物理可行
"""
import argparse
import json
import pathlib

import numpy as np
import mujoco

MENAGERIE = pathlib.Path.home()/"mujoco-lab"/"mujoco_menagerie"/"unitree_g1"/"scene.xml"
FPS = 30.0


class Checker:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.coll = [g for g in range(m.ngeom)
                     if g != self.floor and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.verts = {}
        for g in self.coll:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                self.verts[g] = np.asarray(m.mesh_vert[a:a+n], dtype=np.float64)
        self.link = {"left_foot": "left_ankle_roll_link",
                     "right_foot": "right_ankle_roll_link",
                     "left_hand": "left_wrist_yaw_link",
                     "right_hand": "right_wrist_yaw_link"}
        self.bid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, v)
                    for k, v in self.link.items()}
        self.geom_of = {k: [g for g in self.coll if m.geom_bodyid[g] == b]
                        for k, b in self.bid.items()}

    def _fk(self, q):
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)

    def _low(self, geoms):
        m, d = self.m, self.d
        lo = np.inf
        for g in geoms:
            if g in self.verts:
                R = d.geom_xmat[g].reshape(3, 3)
                lo = min(lo, float((self.verts[g] @ R.T)[:, 2].min() + d.geom_xpos[g][2]))
            else:
                lo = min(lo, float(d.geom_xpos[g][2]) - float(m.geom_size[g][0]))
        return lo

    def _near_kin(self, b1, b2, hops=3):
        m = self.m
        def chain(b):
            out, cur = [], b
            while cur > 0:
                out.append(cur); cur = m.body_parentid[cur]
            return out + [0]
        c1, c2 = chain(b1), chain(b2)
        for i, x in enumerate(c1):
            if x in c2:
                return i + c2.index(x) <= hops
        return False

    def check(self, path, step=4):
        m = self.m
        z = np.load(path, allow_pickle=True)
        q = np.asarray(z["qpos"], dtype=np.float64)
        masks = z["contacts"]
        keys = [str(x) for x in z["contact_keys"]]
        T = len(q)
        r = {"name": pathlib.Path(path).stem, "frames": int(T),
             "seconds": round(T/FPS, 1)}

        # --- 致命项 ---
        r["nan"] = int(np.sum(~np.isfinite(q)))
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        lim = m.jnt_limited[1:].astype(bool)
        j = q[:, 7:]
        over = np.zeros_like(j, bool)
        over[:, lim] = (j[:, lim] < lo[lim]-1e-9) | (j[:, lim] > hi[lim]+1e-9)
        r["joint_violation_pct"] = round(100*float(over.mean()), 4)
        # 瞬移判据用**加速度**而非速度。原来用 15cm/帧(=4.5m/s) 会把所有
        # 跑步冲刺误判——实测 sprint 峰值 6.19m/s 但加速度只有 52.6m/s²（正常），
        # 而真瞬移的 run2_subject1 加速度到 698m/s²。
        # 人类百米世界纪录约 12m/s；人体运动加速度峰值一般 < 50m/s²。
        rv = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) * FPS
        ra = np.linalg.norm(np.diff(q[:, :3], n=2, axis=0), axis=1) * FPS**2
        r["root_speed_max"] = round(float(rv.max()), 2)
        r["root_acc_max"] = round(float(ra.max()), 1)
        r["teleport_frames"] = int(np.sum((rv > 12.0)) + np.sum(ra > 200.0))
        qn = np.linalg.norm(q[:, 3:7], axis=1)
        r["quat_err"] = round(float(np.abs(qn-1).max()), 6)

        # --- 重要项：接触与对地 ---
        idx = np.arange(0, T, step)
        body_low = np.empty(len(idx))
        foot_tilt = {"left_foot": [], "right_foot": []}
        contact_h = {k: [] for k in self.link}
        for c, i in enumerate(idx):
            self._fk(q[i])
            body_low[c] = self._low(self.coll)
            for k in self.link:
                if masks[keys.index(k)][i]:
                    contact_h[k].append(self._low(self.geom_of[k]))
            for k in ("left_foot", "right_foot"):
                if masks[keys.index(k)][i]:
                    R = self.d.xmat[self.bid[k]].reshape(3, 3)
                    foot_tilt[k].append(np.rad2deg(np.arccos(np.clip(R[2, 2], -1, 1))))

        r["penetration_pct"] = round(100*float(np.mean(body_low < -0.01)), 2)
        r["float_pct"] = round(100*float(np.mean(body_low > 0.02)), 2)
        fh = [x for k in ("left_foot", "right_foot") for x in contact_h[k]]
        r["foot_contact_rms_cm"] = round(float(np.sqrt(np.mean(np.square(fh))))*100, 2) if fh else None
        ft = [x for k in foot_tilt for x in foot_tilt[k]]
        r["foot_tilt_med_deg"] = round(float(np.median(ft)), 1) if ft else None
        r["foot_tilt_bad_pct"] = round(100*float(np.mean(np.array(ft) > 45)), 1) if ft else None
        for k in self.link:
            r[f"contact_{k}_pct"] = round(100*float(masks[keys.index(k)].mean()), 1)

        # --- 次要项 ---
        far = 0
        for i in idx:
            self._fk(q[i])
            for c in range(self.d.ncon):
                con = self.d.contact[c]
                if self.floor in (con.geom1, con.geom2) or con.dist >= -0.005:
                    continue
                if not self._near_kin(m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]):
                    far += 1
                    break
        r["selfcol_pct"] = round(100*far/len(idx), 1)
        r["joint_acc"] = round(float(np.mean(np.abs(np.diff(j, n=2, axis=0)))*FPS**2), 2)
        r["joint_vel_p99"] = round(float(np.percentile(np.abs(np.diff(j, axis=0))*FPS, 99)), 2)

        # --- 分级 ---
        fatal, major, minor = [], [], []
        if r["nan"]: fatal.append(f"NaN×{r['nan']}")
        if r["joint_violation_pct"] > 0.01: fatal.append(f"越界{r['joint_violation_pct']}%")
        if r["teleport_frames"]: fatal.append(f"瞬移×{r['teleport_frames']}")
        if r["quat_err"] > 1e-3: fatal.append("四元数")
        if r["penetration_pct"] > 15: major.append(f"穿地{r['penetration_pct']}%")
        if r["float_pct"] > 5: major.append(f"悬空{r['float_pct']}%")
        if r["foot_tilt_bad_pct"] is not None and r["foot_tilt_bad_pct"] > 25:
            major.append(f"脚倾斜{r['foot_tilt_bad_pct']}%")
        if r["foot_contact_rms_cm"] is not None and r["foot_contact_rms_cm"] > 6:
            major.append(f"接触RMS{r['foot_contact_rms_cm']}cm")
        if r["selfcol_pct"] > 8: minor.append(f"自碰撞{r['selfcol_pct']}%")
        if r["joint_acc"] > 25: minor.append(f"抖动{r['joint_acc']}")

        r["fatal"], r["major"], r["minor"] = fatal, major, minor
        r["grade"] = "剔除" if fatal else ("待查" if major else ("可用" if not minor else "可用*"))
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(pathlib.Path.home()/"tools"/"g1_dataset"/"raw"))
    ap.add_argument("--out", default=str(pathlib.Path.home()/"tools"/"g1_dataset"/"manifest.json"))
    a = ap.parse_args()

    files = sorted(pathlib.Path(a.dir).glob("*.npz"))
    print(f"审计 {len(files)} 段\n")
    ck = Checker()
    rows = []
    for i, f in enumerate(files, 1):
        try:
            r = ck.check(f)
        except Exception as e:
            print(f"[{i:>2}/{len(files)}] {f.stem:<40} 失败 {type(e).__name__}")
            continue
        rows.append(r)
        flags = " ".join(r["fatal"] + r["major"] + r["minor"])
        print(f"[{i:>2}/{len(files)}] {r['name']:<40}{r['grade']:<7}{flags}")

    print()
    print("=" * 78)
    from collections import Counter
    c = Counter(r["grade"] for r in rows)
    for g in ("可用", "可用*", "待查", "剔除"):
        if c[g]:
            print(f"  {g:<6}{c[g]:>3} 段"
                  f"   {sum(r['frames'] for r in rows if r['grade']==g):>7} 帧"
                  f"   {sum(r['seconds'] for r in rows if r['grade']==g)/60:>6.1f} 分钟")
    usable = [r for r in rows if r["grade"].startswith("可用")]
    print(f"\n  可用于训练: {len(usable)} 段  "
          f"{sum(r['frames'] for r in usable)} 帧  "
          f"{sum(r['seconds'] for r in usable)/60:.1f} 分钟")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"motions": rows}, ensure_ascii=False, indent=2))
    print(f"\n  manifest -> {out}")


if __name__ == "__main__":
    main()
