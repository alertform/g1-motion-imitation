#!/usr/bin/env python3
"""批量重定向 LAFAN1 全库 -> G1，用同一套参数。

参数已在 7 段差异极大的动作上验证可泛化（走/跑/跳/舞/格斗/瞄准/越障），
所以这里不做逐段调参——真实工程里也不可能，AMASS 有几十万帧。

流水线（与单段版一致）：
    接触检测(高度+速度+迟滞+鞋底朝向) -> mink IK(膝权重30) -> deslip
    -> 限位clip -> 全身逐帧对地

输出 npz，字段: qpos (T,36), fps, contacts (4,T), contact_keys, ground
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time
import traceback

os.environ.update({k: "4" for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

import numpy as np

GMR_ROOT = pathlib.Path.home()/"tools"/"GMR"
sys.path.insert(0, str(GMR_ROOT))
CFG_DIR = GMR_ROOT/"general_motion_retargeting"/"ik_configs"
CFG = CFG_DIR/"bvh_lafan1_to_g1.json"
BACKUP = CFG_DIR/"bvh_lafan1_to_g1.json.orig"


def set_weights(hand_w=50.0, knee_w=30.0):
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
    ap.add_argument("--bvh-dir", default=str(pathlib.Path.home()/"tools"/"lafan1"/"bvh"))
    ap.add_argument("--out-dir", default=str(pathlib.Path.home()/"tools"/"g1_dataset"/"raw"))
    ap.add_argument("--max-frames", type=int, default=0, help="0=全部")
    ap.add_argument("--only", default="", help="只处理名字含此串的")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    if not BACKUP.exists():
        shutil.copy(CFG, BACKUP)

    sys.argv = [sys.argv[0]]                      # 避免下游 argparse 冲突
    import gmr_contact4 as G                       # 复用单段版的全部逻辑
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    bvhs = sorted(pathlib.Path(a.bvh_dir).glob("*.bvh"))
    if a.only:
        bvhs = [p for p in bvhs if a.only.lower() in p.stem.lower()]
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"共 {len(bvhs)} 段待处理 -> {out_dir}")
    print()

    class Args:
        z_thresh, v_thresh, hyst = 0.08, 0.25, 1.6
        ramp, min_seg, root_comp = 4, 6, 0.5
        smooth_window, hand_weight, knee_weight = 7, 50.0, 30.0
        calib_frames = 800
        no_antislip = no_smooth = no_tilt_filter = False
    cfg = Args()

    set_weights(cfg.hand_weight, cfg.knee_weight)
    ok = fail = skip = 0
    t_all = time.perf_counter()
    try:
        for n, p in enumerate(bvhs, 1):
            out = out_dir/f"{p.stem}.npz"
            if a.skip_existing and out.exists():
                skip += 1
                continue
            print(f"[{n:>2}/{len(bvhs)}] {p.stem:<40}", end="", flush=True)
            try:
                t0 = time.perf_counter()
                frames, hh = load_bvh_file(str(p), format="lafan1")
                if a.max_frames:
                    frames = frames[:a.max_frames]

                ground = G.estimate_ground(frames)
                masks = G.detect(frames, 30.0, cfg)
                heights = G.contact_heights(frames, masks)
                weights = {k: G.soft_weight(masks[k], cfg.ramp) for k in masks}
                targets = {k: (None if v is None else v - ground)
                           for k, v in heights.items()}
                anchors = G.planar_anchors(frames, masks, cfg.min_seg)

                probe = G.RobotProbe()
                r = GMR(src_human="bvh_lafan1", tgt_robot="unitree_g1",
                        actual_human_height=hh, verbose=False)

                def run(fr, w, tg, an):
                    return np.asarray([r.retarget(f).copy() for f in
                                       G.apply_constraints(fr, ground, w, tg, an,
                                                           cfg.root_comp)])
                # 闭环标定
                nc = min(cfg.calib_frames, len(frames))
                q1 = run(frames[:nc], {k: v[:nc] for k, v in weights.items()},
                         targets, {k: v[:nc] for k, v in anchors.items()})
                for k in G.END_EFFECTORS:
                    if targets.get(k) is None:
                        continue
                    h = probe.contact_median(q1, weights[k][:nc], k)
                    if h is not None:
                        targets[k] -= h

                qpos = run(frames, weights, targets, anchors)
                qpos[:, 7:] = G.savgol(qpos[:, 7:], cfg.smooth_window)
                qpos[:, 0:3] = G.savgol(qpos[:, 0:3], cfg.smooth_window)

                np.savez_compressed(
                    out, qpos=qpos, fps=np.array(30.0),
                    contacts=np.stack([masks[k] for k in G.END_EFFECTORS]),
                    contact_keys=np.array(list(G.END_EFFECTORS.keys())),
                    ground=np.array(ground))
                print(f" {len(qpos):>5} 帧  {time.perf_counter()-t0:5.1f}s")
                ok += 1
            except Exception as e:
                print(f" 失败: {type(e).__name__}: {str(e)[:50]}")
                fail += 1
    finally:
        shutil.copy(BACKUP, CFG)

    print()
    print(f"完成 {ok}  失败 {fail}  跳过 {skip}   总耗时 "
          f"{(time.perf_counter()-t_all)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
