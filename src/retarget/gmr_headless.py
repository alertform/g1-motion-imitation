#!/usr/bin/env python3
"""GMR 无头批量重定向。

官方的 bvh_to_robot.py 是边解边开窗口渲染的，4742 帧会把 CPU 吃满、
机器变卡；官方的 bvh_to_robot_dataset.py 上游有 import bug
（引用了不存在的 load_lafan1_file）。这个脚本绕开两者：

  - 不开任何窗口，纯求解
  - 用 --threads 限制 BLAS/OMP 线程，默认只用一半核，不跟你抢机器
  - 支持 --max-frames / --stride 先跑小样本验证
  - 输出 npz（不是 pkl），加载不执行代码

用法:
    python gmr_headless.py --bvh <file.bvh> --out <file.npz>
    python gmr_headless.py --bvh <f> --out <o> --max-frames 600 --threads 4
"""
import argparse
import os
import pathlib
import time


def limit_threads(n: int) -> None:
    """必须在 import numpy/mujoco 之前设置才有效。"""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--format", default="lafan1", choices=["lafan1", "nokov"])
    ap.add_argument("--max-frames", type=int, default=0, help="0 = 全部")
    ap.add_argument("--stride", type=int, default=1, help="每隔几帧取一帧")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = 自动取核数的一半")
    a = ap.parse_args()

    if a.threads <= 0:
        a.threads = max(1, (os.cpu_count() or 4) // 2)
    limit_threads(a.threads)

    import numpy as np
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    print(f"线程上限 {a.threads} / 共 {os.cpu_count()} 核")
    print(f"读取 {pathlib.Path(a.bvh).name} ...", flush=True)
    frames, human_height = load_bvh_file(a.bvh, format=a.format)
    total = len(frames)

    sel = list(range(0, total, max(1, a.stride)))
    if a.max_frames:
        sel = sel[:a.max_frames]
    print(f"  源 {total} 帧，实际处理 {len(sel)} 帧（stride={a.stride}）")
    print(f"  推断人体身高 {human_height:.3f} m")

    retargeter = GMR(src_human=f"bvh_{a.format}", tgt_robot=a.robot,
                     actual_human_height=human_height, verbose=False)

    qpos_list = []
    t0 = time.perf_counter()
    last_report = t0
    for k, i in enumerate(sel):
        qpos_list.append(retargeter.retarget(frames[i]).copy())
        now = time.perf_counter()
        if now - last_report > 5.0:
            done = k + 1
            rate = done / (now - t0)
            eta = (len(sel) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(sel)}  {rate:.1f} 帧/秒  剩余约 {eta:.0f}s",
                  flush=True)
            last_report = now

    dt = time.perf_counter() - t0
    qpos = np.asarray(qpos_list)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        qpos=qpos,
        fps=np.array(30.0 / max(1, a.stride)),
        robot=np.array(a.robot),
        source=np.array(pathlib.Path(a.bvh).name),
        human_height=np.array(human_height),
    )
    print(f"完成: {len(qpos)} 帧，用时 {dt:.1f}s（{len(qpos)/dt:.1f} 帧/秒）")
    print(f"  qpos shape = {qpos.shape}")
    print(f"  已保存 {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
