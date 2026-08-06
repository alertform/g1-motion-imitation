#!/usr/bin/env python3
"""重定向数据的系统性质量审计。

覆盖肉眼难以发现、但会毁掉下游 RL 训练的问题：
  1. 数值健全性   NaN/Inf、四元数是否归一化
  2. 关节限位     超限（不只是"接近"）
  3. 关节速度     是否超过硬件上限
  4. 力矩可行性   逆动力学反推所需力矩 vs 电机上限
  5. 自碰撞       身体各部位互相穿模
  6. 足部姿态     接触时脚是平放还是立在边缘
  7. 时间连续性   关节角瞬移
  8. 质心-支撑多边形  静态可行性
  9. 膝关节反曲   物理上不可能的姿态
 10. 与源数据保真度  重定向丢了多少信息
"""
import argparse
import pathlib
from collections import Counter

import numpy as np
import mujoco

FPS = 30.0


class Audit:
    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                       for j in range(m.njnt)]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.issues = []

    def flag(self, sev, title, detail):
        self.issues.append((sev, title, detail))

    # ---------------------------------------------------------------- 1
    def numeric(self, q):
        bad_nan = int(np.sum(~np.isfinite(q)))
        quat = q[:, 3:7]
        norms = np.linalg.norm(quat, axis=1)
        off = np.abs(norms - 1.0)
        print(f"  NaN/Inf 元素: {bad_nan}")
        print(f"  四元数模长: 中位 {np.median(norms):.6f}  最大偏差 {off.max():.6f}")
        if bad_nan:
            self.flag("严重", "数值异常", f"{bad_nan} 个 NaN/Inf")
        if off.max() > 1e-3:
            self.flag("严重", "四元数未归一化",
                      f"最大偏差 {off.max():.4f}，{int(np.sum(off>1e-3))} 帧超标")

    # ---------------------------------------------------------------- 2
    def joint_limits(self, q):
        m = self.m
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        lim = m.jnt_limited[1:].astype(bool)
        j = q[:, 7:]
        over = np.zeros_like(j, bool)
        over[:, lim] = (j[:, lim] < lo[lim] - 1e-6) | (j[:, lim] > hi[lim] + 1e-6)
        pct = 100 * over.mean()
        print(f"  超出限位的 关节-帧: {pct:.3f}%")
        if pct > 0.01:
            worst = np.argsort(-over.mean(axis=0))[:5]
            for k in worst:
                if over[:, k].mean() > 0:
                    exc = np.maximum(lo[k]-j[:, k], j[:, k]-hi[k]).max()
                    print(f"    {self.jnames[k+1]:<28}{100*over[:,k].mean():5.1f}% "
                          f"最大超出 {np.rad2deg(exc):.1f}°")
            self.flag("高", "关节超限", f"{pct:.2f}% 的关节-帧越界")

    # ---------------------------------------------------------------- 3
    def joint_speed(self, q):
        """G1 的关节速度上限：腿约 30 rad/s，臂约 20 rad/s（保守估计）。"""
        v = np.abs(np.diff(q[:, 7:], axis=0)) * FPS
        p99 = np.percentile(v, 99, axis=0)
        mx = v.max(axis=0)
        LIMIT = 20.0
        bad = np.flatnonzero(mx > LIMIT)
        print(f"  关节角速度: 中位 {np.median(v):.2f}  p99 {np.percentile(v,99):.2f} rad/s")
        print(f"  超过 {LIMIT} rad/s 的关节: {len(bad)}")
        for k in bad[:5]:
            print(f"    {self.jnames[k+1]:<28}最大 {mx[k]:6.1f} rad/s  "
                  f"({100*np.mean(v[:,k]>LIMIT):.2f}% 的帧)")
        if len(bad):
            self.flag("高", "关节速度超限",
                      f"{len(bad)} 个关节峰值超 {LIMIT} rad/s，最高 {mx.max():.0f}")

    # ---------------------------------------------------------------- 4
    def torque(self, q, step=3):
        """逆动力学：由 qpos 序列反推所需力矩，与执行器上限比较。"""
        m, d = self.m, self.d
        n = len(q)
        idx = np.arange(1, n-1, step)
        vel = (q[2:, 7:] - q[:-2, 7:]) * (FPS/2)
        acc = (q[2:, 7:] - 2*q[1:-1, 7:] + q[:-2, 7:]) * FPS**2
        need = []
        for c, i in enumerate(idx):
            d.qpos[:] = q[i]
            d.qvel[:] = 0
            d.qvel[6:] = vel[i-1]
            d.qacc[:] = 0
            d.qacc[6:] = acc[i-1]
            mujoco.mj_inverse(m, d)
            need.append(np.abs(d.qfrc_inverse[6:]).copy())
        need = np.array(need)
        # 真正的电机力矩上限在关节的 actuatorfrcrange。
        # 注意不能用 actuator_forcerange——G1 模型里它全是 0（表示未限制），
        # 也不能退回 ctrlrange，那是位置伺服的角度范围（弧度），不是力矩。
        lim = np.zeros(m.nu)
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            r = m.jnt_actfrcrange[j]
            lim[i] = abs(r[1]) if r[1] != 0 else np.inf
        ratio = need / np.maximum(lim, 1e-6)
        print(f"  所需力矩 / 电机上限: 中位 {np.median(ratio):.2f}  "
              f"p95 {np.percentile(ratio,95):.2f}  最大 {ratio.max():.1f}")
        print(f"  超出电机能力的 关节-帧: {100*(ratio > 1.0).mean():.1f}%")
        print()
        print("  【此项仅供参考，不作为问题上报】")
        print("  逆动力学对运动学回放数据无效：qacc 由有限差分得来，与接触约束")
        print("  不自洽（脚可能正“加速穿进地面”），求解器只能算出天文数字的力。")
        print("  实测同一站立姿态：正向仿真真值 0.7 N·m，逆动力学给 99.1 N·m，")
        print("  高估 139 倍。真要评估力矩可行性，得先用轨迹优化让数据物理自洽。")

    # ---------------------------------------------------------------- 5
    def _near_kin(self, b1, b2, hops=3):
        """两个 body 在运动学树上是否相隔 <= hops 跳。

        相邻/近邻连杆的穿透是**模型的碰撞几何裕度问题**，不是姿态错误。
        实测 left_shoulder_roll 在合法范围内只要 < 0.33 rad 就穿躯干，
        最深 11cm——任何垂臂动作都会触发。这类不该算数据问题。
        """
        m = self.m
        def chain(b):
            out, cur = [], b
            while cur > 0:
                out.append(cur)
                cur = m.body_parentid[cur]
            return out + [0]
        c1, c2 = chain(b1), chain(b2)
        for i, x in enumerate(c1):
            if x in c2:
                return i + c2.index(x) <= hops
        return False

    def self_collision(self, q, step=5):
        m, d = self.m, self.d
        cnt_far, cnt_near = Counter(), Counter()
        far_frames = 0
        for i in range(0, len(q), step):
            d.qpos[:] = q[i]
            mujoco.mj_forward(m, d)
            hit_far = False
            for c in range(d.ncon):
                con = d.contact[c]
                if self.floor in (con.geom1, con.geom2) or con.dist >= -0.005:
                    continue
                bid1, bid2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
                b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid1) or "?"
                b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid2) or "?"
                key = tuple(sorted((b1, b2)))
                if self._near_kin(bid1, bid2):
                    cnt_near[key] += 1
                else:
                    cnt_far[key] += 1
                    hit_far = True
            far_frames += hit_far
        total = len(range(0, len(q), step))
        pct = 100*far_frames/total
        print(f"  远端肢体互穿(真问题)的帧: {pct:.1f}%")
        for pair, n in cnt_far.most_common(4):
            print(f"    {pair[0]:<26}{pair[1]:<26}{n:>5} 次")
        if cnt_near:
            print(f"  近邻连杆接触(模型几何裕度，非数据问题): "
                  f"{sum(cnt_near.values())} 次")
            for pair, n in cnt_near.most_common(3):
                print(f"    {pair[0]:<26}{pair[1]:<26}{n:>5} 次")
        if pct > 5:
            self.flag("中", "自碰撞", f"{pct:.0f}% 的帧存在远端肢体互穿")

    # ---------------------------------------------------------------- 6
    def foot_flatness(self, q, masks, keys, step=3):
        """接触时脚底法向应朝上。夹角大 = 用脚尖/脚侧站着。"""
        m, d = self.m, self.d
        for key, link in (("left_foot","left_ankle_roll_link"),
                          ("right_foot","right_ankle_roll_link")):
            if key not in keys:
                continue
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            mask = masks[keys.index(key)]
            angs = []
            for i in range(0, len(q), step):
                if not mask[i]:
                    continue
                d.qpos[:] = q[i]
                mujoco.mj_forward(m, d)
                R = d.xmat[b].reshape(3, 3)
                angs.append(np.rad2deg(np.arccos(np.clip(R[2, 2], -1, 1))))
            if not angs:
                continue
            angs = np.array(angs)
            bad = 100*np.mean(angs > 45)
            print(f"  {key:<12} 接触时脚底倾角: 中位 {np.median(angs):5.1f}°  "
                  f">45° 占 {bad:5.1f}%")
            if bad > 30:
                self.flag("中", "足部姿态", f"{key} {bad:.0f}% 的接触帧脚倾斜超 45°")

    # ---------------------------------------------------------------- 7
    def teleport(self, q):
        dj = np.abs(np.diff(q[:, 7:], axis=0))
        dr = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
        jmax = np.rad2deg(dj.max())
        print(f"  单帧关节最大变化: {jmax:.1f}°   根位置最大位移: {dr.max()*100:.1f} cm")
        big_j = int(np.sum(dj > np.deg2rad(30)))
        big_r = int(np.sum(dr > 0.15))
        print(f"  >30°/帧 的关节变化: {big_j} 次   >15cm/帧 的根位移: {big_r} 次")
        if big_r > 0:
            self.flag("高", "根位置瞬移", f"{big_r} 帧位移超 15cm（=4.5 m/s）")
        if big_j > len(q) * 0.01:
            self.flag("中", "关节瞬移", f"{big_j} 次单帧变化超 30°")

    # ---------------------------------------------------------------- 8
    def com_support(self, q, masks, keys, step=5):
        """静态可行性：质心水平投影是否落在支撑多边形内。

        动态动作可以短暂越界（跑跳都会），但长期在外说明姿态不合理。
        """
        m, d = self.m, self.d
        feet = {}
        for key, link in (("left_foot","left_ankle_roll_link"),
                          ("right_foot","right_ankle_roll_link")):
            feet[key] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
        out_cnt = tot = 0
        dists = []
        for i in range(0, len(q), step):
            pts = [np.asarray(d.xpos[feet[k]][:2]) for k in feet
                   if k in keys and masks[keys.index(k)][i]]
            if len(pts) < 1:
                continue
            d.qpos[:] = q[i]
            mujoco.mj_forward(m, d)
            pts = [np.asarray(d.xpos[feet[k]][:2]).copy() for k in feet
                   if k in keys and masks[keys.index(k)][i]]
            com = np.asarray(d.subtree_com[0][:2])
            center = np.mean(pts, axis=0)
            dist = float(np.linalg.norm(com - center))
            dists.append(dist)
            tot += 1
            if dist > 0.30:                 # 距支撑中心 30cm 以上
                out_cnt += 1
        if tot:
            print(f"  质心距支撑中心: 中位 {np.median(dists)*100:5.1f}cm  "
                  f">30cm 占 {100*out_cnt/tot:5.1f}%")

    # ---------------------------------------------------------------- 9
    def knee_direction(self, q):
        """膝关节应始终为正（向前弯）。负值 = 反曲。"""
        for side in ("left", "right"):
            name = f"{side}_knee_joint"
            if name not in self.jnames:
                continue
            k = self.jnames.index(name) - 1
            v = q[:, 7+k]
            neg = 100*np.mean(v < -0.05)
            print(f"  {name:<24} 范围 [{v.min():+.2f}, {v.max():+.2f}]  "
                  f"反曲(<-0.05) {neg:5.1f}%")
            if neg > 5:
                self.flag("中", "膝关节反曲", f"{name} {neg:.0f}% 的帧为负角")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--xml", default=str(pathlib.Path.home()/"mujoco-lab"/
                                         "mujoco_menagerie"/"unitree_g1"/"scene.xml"))
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    q = np.asarray(z["qpos"], dtype=np.float64)
    masks = z["contacts"] if "contacts" in z else None
    keys = [str(x) for x in z["contact_keys"]] if "contact_keys" in z else []

    au = Audit(a.xml)
    name = pathlib.Path(a.npz).stem
    print("=" * 78)
    print(f"{name}   {len(q)} 帧 @{FPS:g}fps = {len(q)/FPS:.1f}s")
    print("=" * 78)

    for title, fn in [
        ("1. 数值健全性", lambda: au.numeric(q)),
        ("2. 关节限位", lambda: au.joint_limits(q)),
        ("3. 关节速度", lambda: au.joint_speed(q)),
        ("4. 力矩可行性（逆动力学）", lambda: au.torque(q)),
        ("5. 自碰撞", lambda: au.self_collision(q)),
        ("6. 足部姿态", lambda: au.foot_flatness(q, masks, keys) if masks is not None else None),
        ("7. 时间连续性", lambda: au.teleport(q)),
        ("8. 质心-支撑", lambda: au.com_support(q, masks, keys) if masks is not None else None),
        ("9. 膝关节方向", lambda: au.knee_direction(q)),
    ]:
        print(f"\n{title}")
        try:
            fn()
        except Exception as e:
            print(f"  检查失败: {type(e).__name__}: {e}")

    print()
    print("=" * 78)
    print(f"发现 {len(au.issues)} 个问题")
    print("=" * 78)
    for sev, t, det in sorted(au.issues, key=lambda x: {"严重":0,"高":1,"中":2}.get(x[0], 3)):
        print(f"  [{sev}] {t}: {det}")
    if not au.issues:
        print("  未发现明显问题")


if __name__ == "__main__":
    main()
