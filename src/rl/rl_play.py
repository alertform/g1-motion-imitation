#!/usr/bin/env python3
"""在 MuJoCo viewer 里回放训练好的策略。

在 CPU 上跑：策略网络很小，物理用普通 mujoco 就够，这样不会和正在
训练的进程抢显存。

用法:
    python rl_play.py                       # 放最新存档
    python rl_play.py --ckpt runs/walk_v11_final/policy.pkl
    python rl_play.py --check               # 只做一致性自检，不开窗口
"""
import argparse
import os
import pathlib
import time

# 必须在 import jax 之前设：强制 CPU，避免和训练进程抢 GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import mujoco


def build_policy(ckpt, obs_size, act_size):
    """重建网络结构并灌入存档参数。

    brax 存的是纯参数（pytree），网络结构没存，必须用和训练时**完全
    一样**的 network_factory 重建，否则形状对不上。
    """
    import jax_compat  # noqa: F401
    from brax.io import model
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks as ppo_networks

    params = model.load_params(str(ckpt))
    nets = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128))
    make_policy = ppo_networks.make_inference_fn(nets)
    # 训练返回的是 (normalizer, policy, value)，推理只要前两个
    return make_policy(tuple(params[:2]), deterministic=True)


class NumpyRollout:
    """用普通 mujoco 复现 rl_env 的观测/动作约定。

    这里的逻辑和 rl_env 是**重复实现**，两边任何一处改了都必须同步。
    --check 会拿 rl_env 的 jax 版本逐元素对一遍，专门防这个漂移。
    """

    def __init__(self, ref, refv, ctrl_dt=0.02, sim_dt=0.004):
        import rl_env
        self.E = rl_env
        self.refv = np.asarray(refv)                  # 观测里要用参考速度
        # 走 rl_env 的同一个配置函数：求解器、增益全部一致，
        # 否则回放的是另一个物理系统，测出的误差不代表训练时的表现
        self.m = rl_env.configure_model(
            mujoco.MjModel.from_xml_path(str(rl_env.XML)), sim_dt)
        self.d = mujoco.MjData(self.m)
        self.n_frames = int(round(ctrl_dt / sim_dt))
        self.ref = np.asarray(ref)                    # (T, NQ)
        self.T = len(self.ref)
        self.jnt_lo = self.m.jnt_range[1:, 0].copy()
        self.jnt_hi = self.m.jnt_range[1:, 1].copy()
        self.act_scale = rl_env.act_scale(self.m)

    def obs(self, step, last_act):
        q, v = self.d.qpos, self.d.qvel
        w, x, y, z = q[3], q[4], q[5], q[6]
        grav = np.array([-2*(x*z - w*y), -2*(y*z + w*x), -(1 - 2*(x*x + y*y))])
        # 与 rl_env._catch_up / _rot_t 必须逐元素一致
        i = np.clip(step, 0, self.T - 1)
        perr = self.ref[i, :3] - q[:3]
        catch = np.clip(self.E.K_CATCH * perr,
                        -self.E.V_CATCH_MAX, self.E.V_CATCH_MAX)
        vt = self.refv[i, 0:3] + catch
        r = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])
        rt = r.T
        idx = np.clip(step + np.arange(1, self.E.LOOKAHEAD + 1), 0, self.T - 1)
        fut = self.ref[idx, 7:].reshape(-1)
        return np.concatenate([grav, v[3:6], q[7:], v[6:], last_act,
                               rt @ vt, rt @ perr, fut])

    def apply(self, action, step):
        """残差动作：目标 = 下一参考帧关节角 + 有界修正量（与 rl_env 一致）。"""
        ref_jnt = self.ref[min(step + 1, self.T - 1), 7:]
        self.d.ctrl[:] = np.clip(
            ref_jnt + self.act_scale * np.clip(action, -1, 1),
            self.jnt_lo, self.jnt_hi)
        for _ in range(self.n_frames):
            mujoco.mj_step(self.m, self.d)

    def reset(self, step, refv):
        self.d.qpos[:] = self.ref[step]
        self.d.qvel[:] = refv[step]
        mujoco.mj_forward(self.m, self.d)

    def fell(self):
        upright = 1.0 - 2.0 * (self.d.qpos[4]**2 + self.d.qpos[5]**2)
        return self.d.qpos[2] < 0.2 or upright < 0.0


def consistency_check(roll, ref, refv, refb):
    """numpy 观测 vs rl_env 的 jax 观测，逐元素比对。"""
    import jax, jax.numpy as jp
    import rl_env
    env = rl_env.G1Imitate(jp.asarray(ref)[None], jp.asarray(refv)[None],
                           jp.asarray(refb)[None], ep_len=200)
    step = 100
    roll.reset(step, refv)
    last_act = np.zeros(rl_env.NU)
    mine = roll.obs(step, last_act)

    st = env.reset(jax.random.PRNGKey(0))
    st = st.replace(pipeline_state=st.pipeline_state.replace(
        qpos=jp.asarray(roll.d.qpos), qvel=jp.asarray(roll.d.qvel)))
    theirs = np.asarray(env._obs(st.pipeline_state, 0, step, jp.asarray(last_act)))

    err = np.abs(mine - theirs)
    print(f"  观测维度  numpy={mine.shape[0]}  jax={theirs.shape[0]}")
    print(f"  最大逐元素误差 {err.max():.3e}")
    print(f"  {'一致' if err.max() < 1e-4 else '不一致！两边实现已漂移'}")
    return err.max() < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/walk/policy_latest.pkl")
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="回放倍速，1.0 为实时；0.25 便于看清失衡瞬间")
    ap.add_argument("--check", action="store_true", help="只做一致性自检")
    a = ap.parse_args()

    import rl_env
    ref, refv, refb, refc = rl_env.load_reference([a.clip])
    ref, refv = np.asarray(ref[0]), np.asarray(refv[0])
    roll = NumpyRollout(ref, refv)

    if a.check:
        print("=== numpy / jax 观测一致性 ===")
        raise SystemExit(0 if consistency_check(
            roll, ref, refv, np.asarray(refb[0])) else 1)

    ckpt = pathlib.Path(a.ckpt)
    if not ckpt.is_absolute():
        ckpt = pathlib.Path.home()/"tools"/"rl"/ckpt
    if not ckpt.exists():
        raise SystemExit(f"存档不存在: {ckpt}")

    policy = build_policy(ckpt, rl_env.OBS_SIZE, rl_env.NU)
    print(f"载入 {ckpt}   观测 {rl_env.OBS_SIZE} 维")

    import jax
    import mujoco.viewer
    key = jax.random.PRNGKey(0)

    step = a.start
    roll.reset(step, refv)
    last_act = np.zeros(rl_env.NU)

    ctrl_dt = roll.n_frames * roll.m.opt.timestep     # 一个控制步的仿真时长
    print(f"回放速度 {a.speed}×   一步 = {ctrl_dt*1000:.0f}ms 仿真时间")

    with mujoco.viewer.launch_passive(roll.m, roll.d) as v:
        v.cam.distance, v.cam.elevation = 3.0, -12.0
        wall = time.perf_counter()
        while v.is_running():
            o = roll.obs(step, last_act)
            act, _ = policy(o, key)
            act = np.asarray(act)
            roll.apply(act, step)
            last_act = act
            step += 1

            # 实时节流：没有这一段，循环会以 CPU 极限速度推进，
            # 4 秒的仿真被压成 1~2 秒的观感，看起来像「一下就倒了」
            wall += ctrl_dt / max(a.speed, 1e-6)
            lag = wall - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            else:
                wall = time.perf_counter()           # 跟不上就重新对齐，避免累积欠债

            # 相机手动跟随：mjCAMERA_TRACKING 会把 lookat 锁在 body 原点，
            # 趴地姿态时相机会钻到地板下面
            v.cam.lookat[:] = [roll.d.qpos[0], roll.d.qpos[1],
                               max(0.5, roll.d.qpos[2])]
            v.sync()

            if roll.fell() or step >= roll.T - rl_env.LOOKAHEAD - 1:
                n = step - a.start
                print(f"  回合结束：存活 {n} 步 = {n*ctrl_dt:.2f} 秒"
                      f"（{'摔倒' if roll.fell() else '参考播完'}）", flush=True)
                step = a.start
                roll.reset(step, refv)
                last_act = np.zeros(rl_env.NU)
                wall = time.perf_counter()


if __name__ == "__main__":
    main()
