#!/usr/bin/env python3
"""G1 动作模仿的 PPO 训练（Brax + MJX）。

用法:
    python rl_train.py --smoke                    # 冒烟测试，几分钟
    python rl_train.py --clips walk1_subject1     # 单段
    python rl_train.py --grade 可用 --envs 2048   # 按 manifest 筛选
"""
import argparse
import functools
import json
import pathlib
import time

import jax


def pick_clips(grade_filter, names, limit):
    """从 manifest 里挑动作。"""
    mf = pathlib.Path.home()/"tools"/"g1_dataset"/"manifest.json"
    rows = json.loads(mf.read_text())["motions"]
    if names:
        want = [n.strip() for n in names.split(",")]
        rows = [r for r in rows if any(w in r["name"] for w in want)]
    elif grade_filter:
        rows = [r for r in rows if r["grade"].startswith(grade_filter)]
    else:
        rows = [r for r in rows if r["grade"].startswith("可用")]
    rows.sort(key=lambda r: -r["frames"])
    if limit:
        rows = rows[:limit]
    return [r["name"] for r in rows], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="", help="逗号分隔的动作名（子串匹配）")
    ap.add_argument("--grade", default="可用", help="按 manifest 等级筛选")
    ap.add_argument("--limit", type=int, default=8, help="最多用几段")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--ep-len", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟测试")
    ap.add_argument("--out", default=str(pathlib.Path.home()/"tools"/"rl"/"runs"))
    ap.add_argument("--restore", default="",
                    help="从已有存档续训（policy.pkl / policy_latest.pkl）。"
                         "注意 brax 不保存优化器状态和已用步数，"
                         "续训等于用旧权重重新开始计步。")
    a = ap.parse_args()

    if a.smoke:
        a.envs, a.steps, a.limit, a.ep_len = 64, 200_000, 2, 200

    # 必须在 import brax 之前：brax 0.14.2 仍调用 jax 0.10 已移除的
    # device_put_replicated，shim 补回它（见 jax_compat.py）
    import jax_compat  # noqa: F401
    import rl_env
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks

    names, rows = pick_clips(a.grade, a.clips, a.limit)
    print(f"训练用 {len(names)} 段动作:")
    for r in rows:
        print(f"  {r['name']:<32}{r['frames']:>6} 帧  {r['grade']}")
    print()

    ref, refv, refb, refc, refl = rl_env.load_reference(names)
    import numpy as np
    print(f"参考库 shape = {ref.shape}   (段数, 补齐帧数, nq)   已重采样到 50Hz")
    print(f"  真实长度 {int(refl.min())}~{int(refl.max())} 帧，"
          f"合计 {int(refl.sum()):,} 帧 = {refl.sum()*0.02/60:.1f} 分钟")
    print(f"  补齐利用率 {refl.sum()/(len(refl)*ref.shape[1])*100:.1f}%"
          f"（补齐区末帧重复，RSI 与索引均按真实长度）")
    print(f"设备: {jax.devices()}")
    print()

    env = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=a.ep_len)
    eval_env = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=a.ep_len)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    hist = []

    from brax.io import model

    def progress(step, metrics):
        el = time.perf_counter() - t0
        rew = metrics.get("eval/episode_reward", 0.0)
        ln = metrics.get("eval/avg_episode_length", 0.0)
        hist.append({"step": int(step), "reward": float(rew),
                     "ep_len": float(ln), "elapsed": round(el, 1)})
        # 每次评估都落盘：几小时的训练不能只在结尾写一次
        (out/"history.json").write_text(json.dumps(
            {"clips": names, "envs": a.envs, "steps": a.steps,
             "history": hist}, ensure_ascii=False, indent=2))
        print(f"  {int(step):>10} 步   奖励 {float(rew):8.3f}   "
              f"回合长 {float(ln):6.1f}   {el/60:5.1f} 分钟", flush=True)

    def save_ckpt(step, make_policy, params):
        model.save_params(str(out/"policy_latest.pkl"), params)

    make_networks = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        # v4 的教训：默认 init_noise_std=1.0 在 tanh 前空间接近满幅，
        # 探索噪声折合每关节 ±14°+。残差前馈本来就能走 48-81 步，
        # 这么大的噪声把它砸到 33-59 步，训了 6M 步的策略比零动作还差。
        # 0.25 → 探索约 ±7°，够学平衡修正，不至于毁掉前馈。
        init_noise_std=0.25)

    # brax 要求 batch_size * num_minibatches 是 num_envs 的整数倍，
    # 所以 batch_size 得从 num_envs 反推，不能随手填个常数。
    n_minibatch = 4
    batch_size = max(1, a.envs // n_minibatch)
    assert batch_size * n_minibatch % a.envs == 0, \
        f"envs={a.envs} 无法被 {n_minibatch} 个 minibatch 整除"
    print(f"batch_size={batch_size}  num_minibatches={n_minibatch}  envs={a.envs}")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=a.steps,
        num_evals=max(2, a.steps // 2_000_000),
        episode_length=a.ep_len,
        num_envs=a.envs,
        batch_size=batch_size,
        num_minibatches=n_minibatch,
        unroll_length=20,
        num_updates_per_batch=4,
        # brax 默认 0 意味着 env.reset() **整个训练只调用一次**：4096 个
        # 环境的 RSI 起点在开头抽定后永远冻结，AutoResetWrapper 每次都恢复
        # 到 first_pipeline_state。跑 500M 步 = 每个环境把自己那一个起点
        # 重复几万遍，起点分布毫无补充。
        # 设为 1 后每个 eval epoch 重抽一次，500M 步下约 250 次 × 4096 =
        # 100 万次起点采样（原来只有 4096 次）。
        # 见 brax/training/agents/ppo/train.py:823
        num_resets_per_eval=1,
        learning_rate=3e-4,
        # v2 的教训：奖励修好后策略反而在 ~10M 步后崩掉（回合长
        # 35→25→14，但每步回报稳定）——跟踪质量没退化，是更新本身在
        # 破坏平衡能力。两道护栏：
        #   梯度裁剪（brax 默认竟然是 None）
        #   KL 自适应学习率（RSL-RL 同款：KL 超标砍 LR，过低升 LR）
        max_grad_norm=1.0,
        # brax 默认 0.3，腿式运动文献的经验值是 ~0.2。clip 越大，单次更新
        # 允许的策略变化越大——v2 观察到的「更新破坏平衡能力」可能与此有关。
        # 参考: papers/03-训练方法/EmbodimentScalingLaws.pdf
        clipping_epsilon=0.2,
        learning_rate_schedule="ADAPTIVE_KL",
        desired_kl=0.01,
        learning_rate_schedule_min_lr=1e-5,
        learning_rate_schedule_max_lr=1e-3,
        entropy_cost=1e-3,
        # v3 的教训：0.97 的有效视界只有 ~33 步（0.66s），0.7 秒后才摔
        # 的跤几乎不进回报，策略会理性地用「摔得早」换「跟得紧」——
        # 实测就是每步回报稳定、回合长缓跌。0.99 把视界拉到 ~100 步。
        discounting=0.99,
        reward_scaling=1.0,
        normalize_observations=True,
        # 评估用确定性策略：训练曲线要反映真实水平，不是噪声策略的
        # 水平——v4 就是被随机评估掩盖了「越训越差」这个事实
        deterministic_eval=True,
        # 回合到点属于「截断」不是「失败」，需要在价值上自举。这件事由
        # brax 的 EpisodeWrapper 写 info['truncation']、GAE 读它来完成，
        # 一直是开着的。bootstrap_on_timeout 是给自己管时限的环境用的，
        # 要环境手写 info['time_out']——我们不走那条路，别开。
        network_factory=make_networks,
        seed=0,
    )

    restore = None
    if a.restore:
        rp = pathlib.Path(a.restore)
        if not rp.is_absolute():
            rp = pathlib.Path(a.out).parent/rp
        if not rp.exists():
            raise SystemExit(f"续训存档不存在: {rp}")
        restore = tuple(model.load_params(str(rp)))
        print(f"从 {rp} 续训（{len(restore)} 组参数：normalizer/policy/value）")

    print("开始训练…")
    make_policy, params, _ = train_fn(
        environment=env, eval_env=eval_env,
        progress_fn=progress, policy_params_fn=save_ckpt,
        restore_params=restore)

    dt = time.perf_counter() - t0
    print(f"\n训练完成，用时 {dt/60:.1f} 分钟")

    ckpt = out/"policy.pkl"
    model.save_params(str(ckpt), params)
    print(f"  策略 -> {ckpt}")
    print(f"  曲线 -> {out/'history.json'}")


if __name__ == "__main__":
    main()
