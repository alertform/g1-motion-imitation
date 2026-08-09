#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
"""对照实验：12060（崩）vs 3082 / 9246（跑满），悬空量相近。

前四轮改动（阻尼、起点刷新、动作缩放、笛卡尔奖励）都没救回 12060，
「参考数据悬空」也被证伪。这里穷举两组起点在**初始状态**上的差异，
找出唯一区分它们的量。
"""
import numpy as np, mujoco
import rl_env

ref, refv, refb = rl_env.load_reference(["walk1_subject1"])
q, v = np.asarray(ref[0]), np.asarray(refv[0])
m = rl_env.configure_model(mujoco.MjModel.from_xml_path(str(rl_env.XML)))
d = mujoco.MjData(m)

FAIL = [12060]
OK = [3082, 9246, 11558, 12300]

floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
gids = sorted({int(m.pair_geom1[p]) if int(m.pair_geom2[p]) == floor
               else int(m.pair_geom2[p])
               for p in range(m.npair)
               if floor in (int(m.pair_geom1[p]), int(m.pair_geom2[p]))})
FID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
       for n in ("left_ankle_roll_link", "right_ankle_roll_link")]


def probe(f):
    d.qpos[:] = q[f]; d.qvel[:] = v[f]
    mujoco.mj_forward(m, d)
    # 质心与支撑多边形
    com = d.subtree_com[1].copy()
    # 支撑脚：接触面低于 1cm 的
    zs = []
    for g in gids:
        z = d.geom_xpos[g, 2]
        t = m.geom_type[g]
        if t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
            z -= m.geom_size[g, 0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3)
            z -= np.abs(R[2] @ np.diag(m.geom_size[g, :3])).sum()
        zs.append((z, d.geom_xpos[g, :2]))
    ground = [xy for z, xy in zs if z < 0.01]
    if ground:
        pts = np.array(ground)
        # 质心到支撑点集中心的水平距离 + 支撑面跨度
        cop = pts.mean(axis=0)
        com_off = np.linalg.norm(com[:2] - cop)
        span = pts.max(axis=0) - pts.min(axis=0)
    else:
        com_off, span = np.nan, np.array([np.nan, np.nan])
    return dict(
        f=f, ncon=d.ncon, nground=len(ground),
        com_off=com_off, span_x=span[0], span_y=span[1],
        z=q[f, 2],
        tilt=np.degrees(np.arccos(np.clip(1-2*(q[f,4]**2+q[f,5]**2), -1, 1))),
        spd=np.linalg.norm(v[f, 0:3]),
        vz=v[f, 2],
        wnorm=np.linalg.norm(v[f, 3:6]),
        jv_p95=np.percentile(np.abs(v[f, 6:]), 95),
        jv_max=np.abs(v[f, 6:]).max(),
        ke=0.5*np.dot(v[f], v[f]),
        footz_l=d.xpos[FID[0], 2], footz_r=d.xpos[FID[1], 2],
    )


rows = [(f, probe(f), "崩") for f in FAIL] + [(f, probe(f), "满") for f in OK]
keys = [("ncon","接触点"),("nground","触地geom"),("com_off","质心偏移m"),
        ("span_x","支撑跨度x"),("span_y","支撑跨度y"),("z","根高m"),
        ("tilt","倾角°"),("spd","根速度m/s"),("vz","竖直速度m/s"),
        ("wnorm","角速度rad/s"),("jv_p95","关节速p95"),("jv_max","关节速max"),
        ("ke","动能"),("footz_l","左脚z"),("footz_r","右脚z")]

print(f"{'指标':<14}" + "".join(f"{f'{f}({s})':>13}" for f, _, s in rows))
print("-" * (14 + 13*len(rows)))
for k, label in keys:
    line = f"{label:<14}"
    for _, r, _ in rows:
        val = r[k]
        line += f"{val:>13.3f}" if isinstance(val, float) else f"{val:>13}"
    print(line)

print()
print("=== 哪个指标把 12060 与其余四个分开 ===")
fail_v = {k: rows[0][1][k] for k, _ in keys}
for k, label in keys:
    ok_vals = [r[k] for _, r, s in rows if s == "满"]
    lo, hi = min(ok_vals), max(ok_vals)
    fv = fail_v[k]
    if isinstance(fv, float) and np.isnan(fv):
        continue
    if fv < lo or fv > hi:
        rel = (fv - np.mean(ok_vals)) / (np.std(ok_vals) + 1e-9)
        print(f"  {label:<14} 崩={fv:>8.3f}  跑满区间=[{lo:.3f}, {hi:.3f}]  "
              f"偏离 {rel:+.1f}σ")
PYEOF
