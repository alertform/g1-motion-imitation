#!/usr/bin/env python3
"""GMR 重定向：接触约束 + 闭环标定 + 抗抖动 + 防脚滑。

相比 gmr_contact3.py 解决脚滑。实测接触期间机器人的脚在地面水平移动
中位 8~15cm、最大 48cm——而源数据只有 4~6cm，多出来的是重定向引入的。
原因是之前的钉合只约束了**竖直方向**，水平方向完全放任。

新增两层：

  5. 水平钉合   接触段内把该末端的水平位置锁到「段内中位数」。
                用中位数而不是段首：段首那一帧往往还在落地过程中，
                拿它当锚点会把整段拖偏。
                同样用软权重加载，避免在段边界产生冲击。

  6. 根位置补偿  只改末端会让腿被拉长/压缩到 IK 解不动。做法是把
                「所有接触末端的平均水平修正量」反向加到根位置上，
                相当于平移整个人而不是扭曲腿——这才是脚不滑的物理含义
                （脚固定，身体相对脚移动）。

用法:
    python gmr_contact4.py --bvh <f.bvh> --out <f.npz>
    python gmr_contact4.py --bvh <f> --out <o> --no-antislip   # 关掉防滑
"""
import argparse
import json
import os
import pathlib
import shutil
import time

os.environ.update({k: "6" for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

import numpy as np

GMR_ROOT = pathlib.Path.home() / "tools" / "GMR"
CFG_DIR = GMR_ROOT / "general_motion_retargeting" / "ik_configs"
CFG = CFG_DIR / "bvh_lafan1_to_g1.json"
BACKUP = CFG_DIR / "bvh_lafan1_to_g1.json.orig"
MENAGERIE = pathlib.Path.home()/"mujoco-lab"/"mujoco_menagerie"/"unitree_g1"/"scene.xml"

END_EFFECTORS = {"left_foot": "LeftToe", "right_foot": "RightToe",
                 "left_hand": "LeftHand", "right_hand": "RightHand"}
COMOVE = {"left_foot": ["LeftToe", "LeftFoot", "LeftFootMod"],
          "right_foot": ["RightToe", "RightFoot", "RightFootMod"],
          "left_hand": ["LeftHand"], "right_hand": ["RightHand"]}
ROBOT_LINK = {"left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
              "left_hand": "left_wrist_yaw_link", "right_hand": "right_wrist_yaw_link"}
ROOT_BONE = "Hips"


class RobotProbe:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
        self.d = mujoco.MjData(self.m)
        self.geoms, self.body = {}, {}
        for k, link in ROBOT_LINK.items():
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, link)
            if b >= 0:
                self.body[k] = b
                self.geoms[k] = [g for g in range(self.m.ngeom)
                                 if self.m.geom_bodyid[g] == b]

    def _fk(self, qpos):
        self.d.qpos[:] = qpos
        self.mj.mj_forward(self.m, self.d)

    def lowest(self, qpos, key):
        mj, m, d = self.mj, self.m, self.d
        self._fk(qpos)
        lo = np.inf
        for g in self.geoms.get(key, []):
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                R = d.geom_xmat[g].reshape(3, 3)
                lo = min(lo, float(((R @ v.T).T + d.geom_xpos[g])[:, 2].min()))
            else:
                lo = min(lo, float(d.geom_xpos[g][2]) - float(m.geom_size[g][0]))
        return lo

    def xy(self, qpos, key):
        self._fk(qpos)
        return self.d.xpos[self.body[key]][:2].copy()

    def contact_median(self, qpos, w, key, n=150):
        idx = np.flatnonzero(w > 0.5)
        if len(idx) < 10:
            return None
        idx = idx[np.linspace(0, len(idx)-1, min(n, len(idx))).astype(int)]
        return float(np.median([self.lowest(qpos[i], key) for i in idx]))


# ---------------------------------------------------------------- 接触
def hysteresis_mask(z, v, z_on, z_off, v_on, v_off):
    out = np.zeros(len(z), bool)
    state = False
    for i in range(len(z)):
        state = ((z[i] < z_off) and (v[i] < v_off)) if state else \
                ((z[i] < z_on) and (v[i] < v_on))
        out[i] = state
    return out


def soft_weight(mask, ramp):
    w = mask.astype(np.float64)
    if ramp < 1:
        return w
    kern = np.concatenate([np.linspace(0, 1, ramp+1)[1:], [1.0],
                           np.linspace(1, 0, ramp+1)[1:]])
    kern /= kern.sum()
    return np.convolve(np.pad(w, ramp, mode="edge"), kern, mode="valid")[:len(w)]


def segments(mask, min_len=5):
    """返回连续 True 的区间 [(start, end), ...]，end 不含。"""
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


def savgol(y, window, poly=2):
    from scipy.signal import savgol_filter
    window = max(poly + 2, window | 1)
    return y if len(y) <= window else savgol_filter(y, window, poly, axis=0)


def estimate_ground(frames, pct=2.0):
    names = list(frames[0].keys())
    return float(np.percentile([min(f[b][0][2] for b in names) for f in frames], pct))


# 人体脚骨骼的鞋底法向 = 局部 Y 轴。用 walk1 支撑相标定得到：
# 走路支撑相脚必然平放，此时最接近世界 Z 的就是局部 Y（中位 24°，
# 那 24° 是踝-趾骨架的固有偏置，三段动作一致，可作零点）。
FOOT_UP_AXIS = 1
FOOT_TILT_BASE = 25.0        # 零点偏置(度)
FOOT_TILT_TOL = 45.0         # 相对零点允许的偏差


def sole_tilt(frames, side_bone):
    """脚底法向与世界 Z 的夹角(度)。用于剔除「侧躺但不承重」的帧。"""
    from scipy.spatial.transform import Rotation as R
    out = np.empty(len(frames))
    for i, f in enumerate(frames):
        q = np.asarray(f[side_bone][1], dtype=float)      # wxyz
        M = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        out[i] = np.rad2deg(np.arccos(np.clip(M[2, FOOT_UP_AXIS], -1, 1)))
    return out


def detect(frames, fps, a):
    """接触判定：高度低 + 速度低 + （对脚）鞋底朝下。

    第三条是必须的：ground1 这类地面动作里，脚会长时间低速侧躺在地上，
    只看高度和速度会把它判成承重接触。实测宽判据下 80% 的帧被误判，
    脚底倾角中位到了 94°；加上朝向判据后只剩真正的支撑相。
    """
    out = {}
    for key, bone in END_EFFECTORS.items():
        P = np.array([f[bone][0] for f in frames])
        v = np.append(np.linalg.norm(np.diff(P, axis=0), axis=1)*fps, 0.0)
        mask = hysteresis_mask(P[:, 2], v, a.z_thresh, a.z_thresh*a.hyst,
                               a.v_thresh, a.v_thresh*a.hyst)
        if key.endswith("_foot") and not getattr(a, "no_tilt_filter", False):
            side = "Left" if key.startswith("left") else "Right"
            tilt = sole_tilt(frames, f"{side}Foot")
            mask &= np.abs(tilt - FOOT_TILT_BASE) < FOOT_TILT_TOL
        out[key] = mask
    return out


def contact_heights(frames, masks):
    out = {}
    for key, bone in END_EFFECTORS.items():
        mk = masks[key]
        out[key] = (None if mk.sum() < 5 else
                    float(np.median([frames[i][bone][0][2] for i in np.flatnonzero(mk)])))
    return out


def planar_anchors(frames, masks, min_len):
    """每个接触段的水平锚点 = 段内中位数。

    用中位数而非段首：段首那帧常常还在落地过程中，拿它当锚会把整段拖偏。
    返回 {key: (T,2) 锚点数组, 非接触帧为 nan}
    """
    out = {}
    for key, bone in END_EFFECTORS.items():
        P = np.array([f[bone][0][:2] for f in frames])
        anchor = np.full_like(P, np.nan)
        for s, e in segments(masks[key], min_len):
            anchor[s:e] = np.median(P[s:e], axis=0)
        out[key] = anchor
    return out


def apply_constraints(frames, ground, weights, targets, anchors, root_comp):
    """竖直钉合 + 水平钉合 + 根位置补偿。"""
    out = []
    for i, f in enumerate(frames):
        g = {k: [v[0].copy(), v[1].copy()] for k, v in f.items()}
        for b in g:
            g[b][0][2] -= ground

        planar_deltas = []
        for key, w in weights.items():
            wi = float(w[i])
            if wi <= 1e-3 or targets.get(key) is None:
                continue
            bone = END_EFFECTORS[key]
            dz = (targets[key] - g[bone][0][2]) * wi
            dxy = np.zeros(2)
            if anchors is not None and not np.isnan(anchors[key][i]).any():
                dxy = (anchors[key][i] - g[bone][0][:2]) * wi
                planar_deltas.append(dxy)
            for b in COMOVE[key]:
                if b in g:
                    g[b][0][2] += dz
                    g[b][0][:2] += dxy

        # 根位置补偿：整体平移而不是把腿扭长
        if root_comp and planar_deltas and ROOT_BONE in g:
            mean_d = np.mean(planar_deltas, axis=0) * root_comp
            for b in g:
                g[b][0][:2] += mean_d
        out.append(g)
    return out


def set_weights(hand_w, knee_w):
    """调整 IK 位置权重。

    膝关节默认权重是 0(table1)/10(table2)，只有踝的 1/5——IK 只保证踝和
    骨盆到位，膝是顺带解出来的。站立行走时无所谓，但**跪姿时膝就是接触点**，
    误差直接变成肉眼可见的悬空（实测膝误差 6.7cm vs 踝 0.9cm）。

    权重扫描结果（膝误差 / 踝误差 / 骨盆误差）：
        0/10  6.71 / 0.89 / 1.16   默认
        30    5.70 / 0.65 / 1.67   <- 膝改善且踝反而更好
        50    4.54 / 1.50 / 2.34
        80    3.01 / 2.79 / 3.32   膝最好但踝骨盆代价太大
    取 30：踝和骨盆是站立行走的关键，不能为膝牺牲。
    """
    cfg = json.loads(BACKUP.read_text())
    for k in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
        if k in cfg.get("ik_match_table2", {}):
            cfg["ik_match_table2"][k][1] = hand_w
    for k in ("left_knee_link", "right_knee_link"):
        if k in cfg.get("ik_match_table2", {}):
            cfg["ik_match_table2"][k][1] = knee_w
    CFG.write_text(json.dumps(cfg, indent=4))


def retarget_all(prepared, human_h):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    r = GMR(src_human="bvh_lafan1", tgt_robot="unitree_g1",
            actual_human_height=human_h, verbose=False)
    return np.asarray([r.retarget(f).copy() for f in prepared])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--z-thresh", type=float, default=0.08)
    ap.add_argument("--v-thresh", type=float, default=0.25)
    ap.add_argument("--hyst", type=float, default=1.6)
    ap.add_argument("--ramp", type=int, default=4)
    ap.add_argument("--min-seg", type=int, default=6)
    ap.add_argument("--root-comp", type=float, default=0.5,
                    help="根位置补偿系数 0~1，0=不补偿")
    ap.add_argument("--smooth-window", type=int, default=7)
    ap.add_argument("--hand-weight", type=float, default=50.0)
    ap.add_argument("--knee-weight", type=float, default=30.0)
    ap.add_argument("--calib-frames", type=int, default=800)
    ap.add_argument("--no-antislip", action="store_true")
    ap.add_argument("--no-tilt-filter", action="store_true",
                    help="关闭鞋底朝向判据（对照用）")
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()

    if not BACKUP.exists():
        shutil.copy(CFG, BACKUP)
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_h = load_bvh_file(a.bvh, format="lafan1")
    if a.frames:
        frames = frames[:a.frames]
    print(f"源 {len(frames)} 帧")

    ground = estimate_ground(frames)
    masks = detect(frames, 30.0, a)
    heights = contact_heights(frames, masks)
    weights = {k: soft_weight(masks[k], a.ramp) for k in masks}
    targets = {k: (None if v is None else v - ground) for k, v in heights.items()}
    anchors = None if a.no_antislip else planar_anchors(frames, masks, a.min_seg)

    print("接触:")
    for k in END_EFFECTORS:
        segs = segments(masks[k], a.min_seg)
        print(f"  {k:<12} {100*masks[k].mean():5.1f}%   {len(segs):>3} 段")
    print(f"防脚滑: {'关闭' if a.no_antislip else f'开启（根补偿 {a.root_comp:g}）'}")

    probe = RobotProbe()
    set_weights(a.hand_weight, a.knee_weight)
    try:
        n = min(a.calib_frames, len(frames))
        q1 = retarget_all(apply_constraints(
            frames[:n], ground, {k: v[:n] for k, v in weights.items()}, targets,
            None if anchors is None else {k: v[:n] for k, v in anchors.items()},
            a.root_comp), human_h)
        for k in END_EFFECTORS:
            if targets.get(k) is None:
                continue
            h = probe.contact_median(q1, weights[k][:n], k)
            if h is not None:
                targets[k] -= h
        print("标定完成")

        t0 = time.perf_counter()
        qpos = retarget_all(apply_constraints(frames, ground, weights, targets,
                                              anchors, a.root_comp), human_h)
        dt = time.perf_counter() - t0
    finally:
        shutil.copy(BACKUP, CFG)

    if not a.no_smooth:
        qpos[:, 7:] = savgol(qpos[:, 7:], a.smooth_window)
        qpos[:, 0:3] = savgol(qpos[:, 0:3], a.smooth_window)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos, fps=np.array(30.0),
                        contacts=np.stack([masks[k] for k in END_EFFECTORS]),
                        contact_keys=np.array(list(END_EFFECTORS.keys())),
                        ground=np.array(ground))
    print(f"完成 {len(qpos)} 帧，{dt:.1f}s -> {out.name}")

    print("校验:")
    for k in END_EFFECTORS:
        h = probe.contact_median(qpos, weights[k], k)
        segs = segments(masks[k], a.min_seg)
        slips = []
        for s, e in segs:
            P = np.array([probe.xy(qpos[i], k) for i in range(s, e, 2)])
            if len(P) > 1:
                slips.append(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
        hs = f"{h:+.4f}" if h is not None else "  n/a "
        sl = f"{np.median(slips)*100:5.1f}cm" if slips else "  n/a"
        print(f"  {k:<12} 高度 {hs}   段内滑移中位 {sl}")


if __name__ == "__main__":
    main()
