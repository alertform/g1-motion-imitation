#!/usr/bin/env python3
"""对批量重定向的原始输出补跑后处理。

batch_retarget.py 只做到「IK + SavGol」就存盘了，漏掉了单段流水线里的
三步后处理——所以首次审计 77/77 全部因「越界」和「悬空」被剔除。

补上：
    deslip2      防脚滑（腿部低通 + 根位置补偿）
    polish       关节限位 clip + 自碰撞缓解
    ground_body  全身逐帧对地
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path.home()/"tools"))
import deslip2
import polish
import ground_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(pathlib.Path.home()/"tools"/"g1_dataset"/"raw"))
    ap.add_argument("--dst", default=str(pathlib.Path.home()/"tools"/"g1_dataset"/"final"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    dst = pathlib.Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    if a.only:
        files = [f for f in files if a.only.lower() in f.stem.lower()]
    print(f"后处理 {len(files)} 段 -> {dst}\n")

    # ground_body 的 xml 路径是 main() 里用 argparse 传的，不是模块常量
    fk = deslip2.FK()
    probe = ground_body.BodyProbe(polish.MENAGERIE)
    import mujoco
    m = probe.m
    d = mujoco.MjData(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    t_all = time.perf_counter()
    ok = fail = 0
    for i, f in enumerate(files, 1):
        print(f"[{i:>2}/{len(files)}] {f.stem:<40}", end="", flush=True)
        try:
            t0 = time.perf_counter()
            z = np.load(f, allow_pickle=True)
            q = np.asarray(z["qpos"], dtype=np.float64)
            masks, keys = z["contacts"], [str(x) for x in z["contact_keys"]]

            # 1) 防脚滑：两轮「腿部低通 + 根位置补偿」
            for _ in range(2):
                q = deslip2.smooth_legs_in_contact(q, masks, keys, fk, 11, 6)
                q = deslip2.compensate_root(q, masks, keys, fk, 6, 9)

            # 2) 限位 clip + 自碰撞缓解
            q, _ = polish.clip_limits(q, m)
            q = polish.relax_collisions(q, m, d, floor, window=9, passes=2)
            q, _ = polish.clip_limits(q, m)

            # 3) 全身逐帧对地
            q, _ = ground_body.ground_body(q, probe, 5, 11, 0.005)

            np.savez_compressed(dst/f.name, qpos=q, fps=z["fps"],
                                contacts=masks, contact_keys=z["contact_keys"],
                                ground=z["ground"])
            print(f" {len(q):>5} 帧  {time.perf_counter()-t0:5.1f}s")
            ok += 1
        except Exception as e:
            print(f" 失败 {type(e).__name__}: {str(e)[:50]}")
            fail += 1

    print(f"\n完成 {ok}  失败 {fail}   总耗时 {(time.perf_counter()-t_all)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
