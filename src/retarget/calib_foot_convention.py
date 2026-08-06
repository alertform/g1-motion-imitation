#!/usr/bin/env python3
"""用 walk1 标定人体脚骨骼的坐标系约定，再据此判断其他动作里脚是否侧躺。

原理：正常行走的支撑相，脚必然平放在地上。此时脚骨骼局部坐标系里
「朝上」的那个轴就是鞋底法向。标定出来之后，就能用同一个轴去衡量
ground1 这类动作里脚到底是平放还是侧着。
"""
import os
os.environ.update({k: "6" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.utils.lafan1 import load_bvh_file

BVH = pathlib.Path.home()/"tools"/"lafan1"/"bvh"


def load(name, n=2000):
    frames, _ = load_bvh_file(str(BVH/f"{name}.bvh"), format="lafan1")
    frames = frames[:n]
    names = list(frames[0].keys())
    ground = np.percentile([min(f[b][0][2] for b in names) for f in frames], 2)
    return frames, ground


def stance_mask(frames, ground, side):
    toe = np.array([f[f"{side}Toe"][0] for f in frames])
    spd = np.append(np.linalg.norm(np.diff(toe, axis=0), axis=1)*30, 0)
    return (toe[:, 2] - ground < 0.05) & (spd < 0.15)


def axis_angles(frames, mask, side):
    """返回脚骨骼三个局部轴（及其反向）与世界 Z 的夹角中位数。"""
    out = np.zeros((3, 2))
    for ax in range(3):
        vals = [[], []]
        for i in np.flatnonzero(mask):
            q = np.asarray(frames[i][f"{side}Foot"][1], float)   # wxyz
            M = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            v = M[:, ax]
            vals[0].append(np.rad2deg(np.arccos(np.clip(v[2], -1, 1))))
            vals[1].append(np.rad2deg(np.arccos(np.clip(-v[2], -1, 1))))
        out[ax] = [np.median(vals[0]), np.median(vals[1])]
    return out


print("=" * 74)
print("第 1 步：用 walk1 支撑相标定人体脚骨骼的「朝上」轴")
print("=" * 74)
frames, g = load("walk1_subject1")
best = None
for side in ("Left", "Right"):
    mk = stance_mask(frames, g, side)
    A = axis_angles(frames, mk, side)
    print(f"  {side}脚 支撑相 {mk.sum()} 帧")
    for ax in range(3):
        for sgn, lab in ((0, ""), (1, "(取反)")):
            print(f"    局部{'XYZ'[ax]}{lab:<6} 与世界Z夹角中位 {A[ax][sgn]:6.1f}°")
    flat = np.unravel_index(np.argmin(A), A.shape)
    print(f"    -> 最接近 0° 的是 局部{'XYZ'[flat[0]]}"
          f"{'(取反)' if flat[1] else ''}  = {A[flat]:.1f}°")
    if best is None:
        best = flat
    print()

AX, SGN = best
print(f"  标定结果：人体脚骨骼的鞋底法向 = 局部 {'XYZ'[AX]} 轴"
      f"{'（取反）' if SGN else ''}")

print()
print("=" * 74)
print("第 2 步：用它衡量各动作接触帧里脚的侧倾")
print("=" * 74)
for name in ("walk1_subject1", "fallAndGetUp1_subject1", "ground1_subject1"):
    p = BVH/f"{name}.bvh"
    if not p.exists():
        continue
    frames, g = load(name)
    print(f"  {name}")
    for side in ("Left", "Right"):
        mk = stance_mask(frames, g, side)
        if mk.sum() < 10:
            print(f"    {side}: 接触帧不足")
            continue
        angs = []
        for i in np.flatnonzero(mk):
            q = np.asarray(frames[i][f"{side}Foot"][1], float)
            M = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            v = M[:, AX] * (-1 if SGN else 1)
            angs.append(np.rad2deg(np.arccos(np.clip(v[2], -1, 1))))
        angs = np.array(angs)
        print(f"    {side}脚 接触 {mk.sum():>4} 帧   鞋底法向倾角 中位 {np.median(angs):6.1f}°"
              f"   >45° 占 {100*np.mean(angs>45):5.1f}%")
    print()

print("=" * 74)
print("判读：walk1 应接近 0°（脚平放）。若 ground1 也小，说明源数据脚是平的，")
print("      机器人 94° 就是重定向的错；若 ground1 也大，则是忠实还原。")
