#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
import numpy as np, mujoco
import rl_env

ref, refv = rl_env.load_reference(["walk1_subject1"])
q, v = np.asarray(ref[0]), np.asarray(refv[0])
T = len(q)
print(f"walk1_subject1: {T} 帧 @50Hz = {T*0.02:.1f} 秒")
print()

# 评估用的 16 个起点（与 rl_eval_mjx 一致）
starts = np.linspace(0, T - 1000 - rl_env.LOOKAHEAD - 2, 16).astype(int)
FAIL = {11: 446, 15: 102}     # 评估里没跑满的两个（下标, 步数）

print(f"  {'#':>3} {'起点帧':>8} {'时刻':>8} {'速度m/s':>9} {'转向°/s':>9} "
      f"{'根高cm':>8} {'膝角°':>8}  20秒结果")
m = mujoco.MjModel.from_xml_path(str(rl_env.XML))
kj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
kq = m.jnt_qposadr[kj]

for i, s in enumerate(starts):
    w = slice(s, min(s+1000, T))
    sp = np.linalg.norm(v[w, 0:3], axis=1)
    yaw = np.abs(v[w, 5])                      # 机体系 yaw 角速度
    res = f"{FAIL[i]} 步 ✗" if i in FAIL else "跑满 ✓"
    print(f"  {i:>3} {s:>8} {s*0.02:>7.1f}s {sp.mean():>9.2f} "
          f"{np.degrees(yaw.mean()):>9.1f} {q[w,2].mean()*100:>8.1f} "
          f"{np.degrees(q[w,kq]).mean():>8.1f}  {res}")

print()
print("=== 失败窗口 vs 成功窗口的差异 ===")
def stats(idxs):
    sp, yaw, acc = [], [], []
    for i in idxs:
        w = slice(starts[i], min(starts[i]+1000, T))
        sp.append(np.linalg.norm(v[w,0:3],axis=1))
        yaw.append(np.abs(v[w,5]))
        acc.append(np.abs(np.diff(v[w,0:3],axis=0)).max(axis=1)/0.02)
    return np.concatenate(sp), np.concatenate(yaw), np.concatenate(acc)

sf, yf, af = stats(list(FAIL))
so, yo, ao = stats([i for i in range(16) if i not in FAIL])
print(f"  {'':<14}{'失败窗口':>12}{'成功窗口':>12}")
print(f"  {'平均速度':<14}{sf.mean():>11.2f}{so.mean():>12.2f} m/s")
print(f"  {'速度p95':<14}{np.percentile(sf,95):>11.2f}{np.percentile(so,95):>12.2f} m/s")
print(f"  {'转向速率均值':<12}{np.degrees(yf.mean()):>11.1f}{np.degrees(yo.mean()):>12.1f} °/s")
print(f"  {'转向p95':<14}{np.degrees(np.percentile(yf,95)):>11.1f}{np.degrees(np.percentile(yo,95)):>12.1f} °/s")
print(f"  {'加速度p95':<13}{np.percentile(af,95):>11.1f}{np.percentile(ao,95):>12.1f} m/s²")
PYEOF
