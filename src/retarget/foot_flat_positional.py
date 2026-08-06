#!/usr/bin/env python3
"""用位置（而非朝向）判断人体的脚是否平放——不依赖坐标系约定。

判据：脚踝->脚趾 向量与水平面的夹角。
  平放      -> 接近 0°
  踮脚/绷直  -> 明显为负（脚趾更低）
  脚跟着地   -> 明显为正
侧躺时脚踝与脚趾高度接近，仍会显示接近 0——所以再补一个判据：
脚踝相对地面的高度。平放站立时脚踝约 7cm，侧躺时会低得多。
"""
import os
os.environ.update({k: "6" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
import pathlib
import numpy as np
from general_motion_retargeting.utils.lafan1 import load_bvh_file

for name in ("ground1_subject1", "fallAndGetUp1_subject1", "walk1_subject1"):
    p = pathlib.Path.home()/"tools"/"lafan1"/"bvh"/f"{name}.bvh"
    if not p.exists():
        continue
    frames, _ = load_bvh_file(str(p), format="lafan1")
    frames = frames[:2000]
    names = list(frames[0].keys())
    ground = np.percentile([min(f[b][0][2] for b in names) for f in frames], 2)

    print("=" * 72)
    print(name)
    print("=" * 72)
    for side in ("Left", "Right"):
        ank = np.array([f[f"{side}Foot"][0] for f in frames])
        toe = np.array([f[f"{side}Toe"][0] for f in frames])
        ank[:, 2] -= ground; toe[:, 2] -= ground
        v = toe - ank
        horiz = np.linalg.norm(v[:, :2], axis=1)
        pitch = np.rad2deg(np.arctan2(v[:, 2], np.maximum(horiz, 1e-6)))

        # 只看「低且慢」的帧，也就是我的接触检测会命中的那些
        spd = np.append(np.linalg.norm(np.diff(toe, axis=0), axis=1)*30, 0)
        mask = (toe[:, 2] < 0.08) & (spd < 0.25)

        if mask.sum() < 10:
            print(f"  {side}: 接触帧太少")
            continue
        print(f"  {side}脚  接触帧 {mask.sum()}/{len(frames)} ({100*mask.mean():.0f}%)")
        print(f"    脚踝->脚趾 俯仰角: 中位 {np.median(pitch[mask]):+6.1f}°  "
              f"|>30°| 占 {100*np.mean(np.abs(pitch[mask])>30):5.1f}%")
        print(f"    脚踝离地高度     : 中位 {np.median(ank[mask,2])*100:5.1f}cm  "
              f"（平放站立约 7cm）")
        low_ank = 100*np.mean(ank[mask, 2] < 0.04)
        print(f"    脚踝低于 4cm 的帧: {low_ank:5.1f}%   <- 高说明脚是侧躺不是平放")
    print()

print("=" * 72)
print("判读")
print("=" * 72)
print("  walk1 是正常行走，可作基准：脚踝高度应在 7cm 左右、俯仰角小。")
print("  若 ground1 的脚踝高度明显更低，说明**源数据里脚本来就侧躺在地上**，")
print("  机器人跟着侧躺是忠实的——问题在我的接触检测把它当成了承重接触。")
