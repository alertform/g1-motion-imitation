#!/usr/bin/env python3
"""GMR + 接触硬约束：把接触末端做成 IK 的等式约束，而不是软目标。

之前所有版本都是「调权重 + 事后修正」。权重是软的——IK 会在膝、踝、骨盆、
手之间做加权折中，谁也不保证。实测膝权重扫描的取舍很残酷：
    膝权重 30 -> 膝误差 5.7cm
    膝权重 80 -> 膝误差 3.0cm，但踝误差从 0.9 涨到 2.8cm

mink 的 solve_ik 有独立的 `constraints` 参数（不是 `limits`），接受一组
Task 作为**等式硬约束**：求解器必须满足它们，其余 task 只能在剩余零空间里
尽力。这正是接触该有的语义——脚在地上就是在地上，不是"尽量在"。

做法：
  - 接触帧：把该末端的 FrameTask 移到 constraints
  - 非接触帧：留在普通 tasks 里
  - 每帧动态切换

用法:
    python gmr_hard.py --bvh <f.bvh> --out <f.npz>
    python gmr_hard.py --bvh <f> --out <o> --no-hard   # 对照：全软约束
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
# 接触时要作为硬约束的机器人 frame。膝也纳入——跪姿时它是接触点。
HARD_FRAMES = {
    "left_foot": ["left_ankle_roll_link"],
    "right_foot": ["right_ankle_roll_link"],
    "left_hand": ["left_wrist_yaw_link"],
    "right_hand": ["right_wrist_yaw_link"],
}
FOOT_UP_AXIS = 1
FOOT_TILT_BASE = 25.0
FOOT_TILT_TOL = 45.0


def hysteresis_mask(z, v, z_on, z_off, v_on, v_off):
    out = np.zeros(len(z), bool)
    st = False
    for i in range(len(z)):
        st = ((z[i] < z_off) and (v[i] < v_off)) if st else \
             ((z[i] < z_on) and (v[i] < v_on))
        out[i] = st
    return out


def sole_tilt(frames, bone):
    from scipy.spatial.transform import Rotation as R
    out = np.empty(len(frames))
    for i, f in enumerate(frames):
        q = np.asarray(f[bone][1], dtype=float)
        M = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        out[i] = np.rad2deg(np.arccos(np.clip(M[2, FOOT_UP_AXIS], -1, 1)))
    return out


def detect(frames, fps, a):
    out = {}
    for key, bone in END_EFFECTORS.items():
        P = np.array([f[bone][0] for f in frames])
        v = np.append(np.linalg.norm(np.diff(P, axis=0), axis=1)*fps, 0.0)
        mask = hysteresis_mask(P[:, 2], v, a.z_thresh, a.z_thresh*a.hyst,
                               a.v_thresh, a.v_thresh*a.hyst)
        if key.endswith("_foot"):
            side = "Left" if key.startswith("left") else "Right"
            tilt = sole_tilt(frames, f"{side}Foot")
            mask &= np.abs(tilt - FOOT_TILT_BASE) < FOOT_TILT_TOL
        out[key] = mask
    return out


def estimate_ground(frames, pct=2.0):
    names = list(frames[0].keys())
    return float(np.percentile([min(f[b][0][2] for b in names) for f in frames], pct))


def patch_retarget(retargeter, hard_frames_per_frame):
    """给 GMR 实例打补丁，让 retarget() 支持每帧指定硬约束。

    不改 GMR 源码，只在实例上换掉方法——避免污染仓库。
    """
    import mink

    # frame_name -> task 的索引
    name_to_task2 = {}
    for fname, entry in retargeter.ik_match_table2.items():
        body_name = entry[0]
        t = retargeter.human_body_to_task2.get(body_name)
        if t is not None:
            name_to_task2[fname] = t

    def retarget_hard(human_data, hard_names=(), offset_to_ground=False):
        retargeter.update_targets(human_data, offset_to_ground)
        dt = retargeter.configuration.model.opt.timestep

        # 第一遍：原样（table1 负责躯干和大关节的粗对齐）
        if retargeter.use_ik_match_table1:
            for _ in range(retargeter.max_iter):
                vel = mink.solve_ik(retargeter.configuration, retargeter.tasks1,
                                    dt, retargeter.solver, retargeter.damping,
                                    limits=retargeter.ik_limits)
                retargeter.configuration.integrate_inplace(vel, dt)

        # 第二遍：把接触末端的 task 抽出来当硬约束
        if retargeter.use_ik_match_table2:
            hard = [name_to_task2[n] for n in hard_names if n in name_to_task2]
            soft = [t for t in retargeter.tasks2 if t not in hard]
            for _ in range(retargeter.max_iter):
                try:
                    vel = mink.solve_ik(
                        retargeter.configuration, soft, dt, retargeter.solver,
                        retargeter.damping, limits=retargeter.ik_limits,
                        constraints=hard if hard else None)
                except Exception:
                    # 硬约束无解（目标物理上够不到）时退回全软，保证不中断
                    vel = mink.solve_ik(
                        retargeter.configuration, retargeter.tasks2, dt,
                        retargeter.solver, retargeter.damping,
                        limits=retargeter.ik_limits)
                retargeter.configuration.integrate_inplace(vel, dt)

        return retargeter.configuration.data.qpos.copy()

    return retarget_hard


def set_weights(hand_w, knee_w):
    cfg = json.loads(BACKUP.read_text())
    for k in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
        if k in cfg.get("ik_match_table2", {}):
            cfg["ik_match_table2"][k][1] = hand_w
    for k in ("left_knee_link", "right_knee_link"):
        if k in cfg.get("ik_match_table2", {}):
            cfg["ik_match_table2"][k][1] = knee_w
    CFG.write_text(json.dumps(cfg, indent=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--z-thresh", type=float, default=0.08)
    ap.add_argument("--v-thresh", type=float, default=0.25)
    ap.add_argument("--hyst", type=float, default=1.6)
    ap.add_argument("--hand-weight", type=float, default=50.0)
    ap.add_argument("--knee-weight", type=float, default=30.0)
    ap.add_argument("--ramp", type=int, default=3,
                    help="接触段边界腐蚀帧数，避免硬约束突然锁死")
    ap.add_argument("--no-hard", action="store_true", help="对照组：不用硬约束")
    a = ap.parse_args()

    if not BACKUP.exists():
        shutil.copy(CFG, BACKUP)
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, hh = load_bvh_file(a.bvh, format="lafan1")
    if a.frames:
        frames = frames[:a.frames]
    print(f"源 {len(frames)} 帧")

    ground = estimate_ground(frames)
    masks = detect(frames, 30.0, a)
    for k in END_EFFECTORS:
        print(f"  {k:<12} 接触 {100*masks[k].mean():5.1f}%")

    # 人体数据整体对地
    prepared = []
    for f in frames:
        g = {k: [v[0].copy(), v[1].copy()] for k, v in f.items()}
        for b in g:
            g[b][0][2] -= ground
        prepared.append(g)

    # 腐蚀接触掩码，得到「稳定核心」
    def erode(mask, k):
        if k < 1:
            return mask.copy()
        out = mask.copy()
        for i in range(len(mask)):
            lo_i, hi_i = max(0, i-k), min(len(mask), i+k+1)
            out[i] = mask[lo_i:hi_i].all()
        return out

    core = {k: erode(v, a.ramp) for k, v in masks.items()}
    for k in END_EFFECTORS:
        print(f"  {k:<12} 硬约束核心 {100*core[k].mean():5.1f}% "
              f"(接触 {100*masks[k].mean():.1f}%)")

    set_weights(a.hand_weight, a.knee_weight)
    try:
        r = GMR(src_human="bvh_lafan1", tgt_robot="unitree_g1",
                actual_human_height=hh, verbose=False)
        rt = patch_retarget(r, None)
        t0 = time.perf_counter()
        qs = []
        n_hard = 0
        for i, f in enumerate(prepared):
            if a.no_hard:
                names = ()
            else:
                # 只在接触段的「稳定核心」施加硬约束：段边界前后 ramp 帧不加。
                # 硬约束是二值的（要么锁死要么不管），在切换帧突然锁死会让
                # IK 剧烈调整关节——实测关节加速度从 8.06 涨到 12.38。
                # 退到核心区可以保住接触精度，同时把冲击留给软约束平滑接管。
                names = [n for key, mk in core.items() if mk[i]
                         for n in HARD_FRAMES[key]]
                n_hard += len(names) > 0
            qs.append(rt(f, hard_names=names))
        qpos = np.asarray(qs)
        dt = time.perf_counter() - t0
    finally:
        shutil.copy(BACKUP, CFG)

    print(f"完成 {len(qpos)} 帧，{dt:.1f}s ({len(qpos)/dt:.0f} 帧/秒)"
          f"   使用硬约束的帧 {100*n_hard/len(qpos):.0f}%")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos, fps=np.array(30.0),
                        contacts=np.stack([masks[k] for k in END_EFFECTORS]),
                        contact_keys=np.array(list(END_EFFECTORS.keys())),
                        ground=np.array(ground))
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
