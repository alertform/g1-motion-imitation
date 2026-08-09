#!/usr/bin/env python3
"""G1 动作模仿环境（MJX + Brax PPO）。

DeepMimic 范式：策略学的是「如何在真实物理下逼近参考动作」，
不是照搬参考动作。所以参考动作**不需要物理可行**——这正是我们
之前审计时把「力矩可行性」判定为无效指标的原因。

观测（只用真机拿得到的量，否则 sim2real 必翻车）：
    重力在机身系的投影   3    等价于 roll/pitch，来自 IMU
    机身角速度           3    陀螺仪
    关节角               29   编码器
    关节角速度           29   编码器差分
    上一步动作           29
    参考动作的未来 K 帧   K×29  相位信息
奖励：
    关节角相似度 + 根位置 + 根姿态 + 存活 - 能耗
"""
import functools
import pathlib
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

XML = pathlib.Path.home()/"mujoco-lab"/"mujoco_menagerie"/"unitree_g1"/"scene_mjx.xml"
DATASET = pathlib.Path.home()/"tools"/"g1_dataset"/"final"

NU = 29                      # 执行器数
NQ = 36                      # 3 位置 + 4 四元数 + 29 关节
NV = 35                      # 6 根自由度 + 29 关节
LOOKAHEAD = 4                # 观测里包含参考动作的未来帧数

# 观测维度：重力3 + 根角速度3 + 关节角29 + 关节角速度29 + 上一步动作29
#           + 目标根速度(机身系)3 + 根位置误差(机身系)3 + 未来 K 帧关节角 4×29
# 定成常量而不是让三个脚本各算一遍——之前求解器设置就是各写一份而漂移过
OBS_SIZE = 3 + 3 + NU + NU + NU + 3 + 3 + LOOKAHEAD * NU

# 奖励的衰减系数。选取原则：在**关心的误差区间内**要有梯度。
# exp(-k·e) 一旦饱和到 1e-3，策略就分不出 40° 和 20° 的好坏了。
K_POSE = 6.0                 # 0.50 分 @ 20° RMS，0.90 分 @ 8°
K_ROOT = 20.0                # 0.50 分 @ 19cm，0.90 分 @ 7cm
K_ORIENT = 40.0              # 0.50 分 @ 21°，0.90 分 @ 9°
K_RVEL = 8.0                 # 0.50 分 @ 0.29m/s 误差，0.90 分 @ 0.11m/s
K_JVEL = 0.2                 # 0.50 分 @ 1.86rad/s RMS

# 位置回正：目标速度 = 参考速度 + K_CATCH·(参考位置 - 当前位置)，回正项限幅。
# v8 的教训：只跟踪参考速度消不掉已积累的位置误差——位置是速度的积分，
# 速度跟得再准，早先漂掉的那部分也不会自己回来。实测根速度误差从 0.948
# 压到 0.481 m/s（-49%），根漂移却纹丝不动（24.81 -> 24.44cm）。
# 更麻烦的是这两项本来在互相打架：要补回位置就必须偏离参考速度，而
# r_rvel 恰好惩罚这种偏离。改成带回正项的目标速度后，跟踪它等价于一个
# P 控制器，会主动把位置误差拉回零，冲突消失。
K_CATCH = 1.0                # 1/s，24cm 误差 -> 0.24m/s 追赶速度
V_CATCH_MAX = 0.5            # m/s，限幅避免大误差时目标速度不可达

# ---- 笛卡尔身体跟踪（替代关节角跟踪）----
# 关节角误差**不区分杠杆**：髋差 10° 让整个身体位移十几厘米，腕差 10° 只动
# 几厘米，但在 29 关节的均值里权重相同。深蹲姿态（frame 12060）恰恰是膝髋
# 小角度偏差造成大空间位移的场景——v11~v13 三轮改动（阻尼、起点刷新、
# per-joint 动作缩放）都没能救回它，跟踪先失效（21 步内关节误差 1.2°→8.2°）
# 然后才失稳，指向奖励结构本身。
# BeyondMimic 的做法：只跟踪笛卡尔空间的身体位姿，不设关节角项；
# 全局位置/朝向由 anchor（骨盆）两项管，其余 body 跟踪**相对 anchor** 的
# 位姿——这样「机器人在哪」和「摆成什么形状」被解耦。
# 参考 papers/06-最新进展/BeyondMimic_GuidedDiffusion.pdf
ANCHOR_BODY = "pelvis"
TRACK_BODIES = (
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "torso_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
K_BODY = 44.4                # = 1/0.15²，0.90 分 @ 5cm，0.37 分 @ 15cm
MAX_BODY_ERR = 0.09          # m²（均值），约 30cm RMS —— 早停阈值

# RSI 起点过滤：MuJoCo 接触点（d.ncon）少于该数的帧不能作为回合起点。
# 依据（对照实验，diag-contrast.sh）：frame 12060 四轮改动（阻尼、起点
# 刷新、动作缩放、笛卡尔奖励）都救不回来，穷举初始状态后唯一以 -4.0σ
# 区分成败的指标是 **d.ncon = 1**——支撑面退化成一个点。
# 注意不能用「表面低于阈值的 geom 数」代替：12060 有 2 个 geom 贴地，
# 但都在同一只脚上（支撑跨度仅 2×5cm），geom 计数会漏掉这种单点支撑
# ——第一版实现就犯了这个错，被验证脚本抓出来。
# 单点支撑的瞬间被 RSI「凭空放置」，没有此前积累的动量与接触历史，
# 任何扰动都无法靠调整压力中心抵抗——这不是策略能力问题，是该帧
# 物理上就不适合作为静态初始化的起点。
MIN_START_CONTACTS = 2

# 奖励权重（正项之和 = 1.0）。v7 只有位姿+根位置+朝向，缺速度项——
# 位置误差会累积且不可逆：速度稍偏一点，位置就一路漂走再也回不来。
# v7 实测根漂移中位 41cm，就是这么来的。
# v14：主项从关节角改为笛卡尔身体位置。关节角保留 0.10 作弱正则，
# 防止「空间位置对但关节配置离谱」的解（IK 多解 / 镜像）。
W_BODY = 0.30
W_POSE, W_ROOT, W_ORIENT = 0.10, 0.12, 0.08
W_RVEL, W_JVEL, W_ALIVE = 0.20, 0.05, 0.15

# 早停阈值：跟丢多少就判定这个回合失败
MAX_POSE_ERR = 0.6           # rad²（均值），约 44° RMS
MAX_ROOT_ERR = 0.36          # m²，即 60cm

# 残差动作幅度：ctrl = 参考关节角 + ACT_SCALE·act。
# v3 的教训：让策略从零输出绝对目标角，它必须每步「背出」整条轨迹，
# 而输出 0 对应的是关节限位中点这种大字站姿。残差化之后输出 0 就是
# 前馈跟踪参考，RL 只需要学修正量。真机上参考轨迹同样可得，不伤 sim2real。
# 残差幅度改为 per-joint，不再用一个全局常数。
#
# 思路来自 BeyondMimic：幅度应正比于关节的力矩能力（踝 50 N·m vs 膝 139 N·m
# 本就该有不同的修正范围）。但**不能照搬它的绝对公式** `0.25×力矩/kp`——
# 那配的是他们 kp=14~99 的低增益，我们 kp=250 高 3~17 倍，同一公式算出
# 0.29°~7.96°，实测比策略实际用量（p95 15~17°）小 13.6 倍，腕关节差 58 倍，
# 等于把控制权威砍掉。
#
# 改为：保留 per-joint 的**相对比例**（∝ √力矩，避免腕/髋差 28 倍过于悬殊），
# 再整体标定到 ACT_SCALE_REF，使中位数与原全局值相当。
# 同时用行程做上限保护——踝 roll 行程仅 ±15°，原全局 0.3 rad 是行程的 115%，
# 实测右踝截断率 20%（全关节平均仅 1.3%，所以这不是主要问题，但该修）。
ACT_SCALE_REF = 0.3          # 标定基准：per-joint 幅度的中位数
ACT_RANGE_FRAC = 0.6         # 上限保护：不超过关节半行程的这个比例

# 位置伺服增益。**这是 v1~v5 全部失败的根因**：
# g1_mjx.xml 号称「改用更低、更真实的 PD 增益」（kp=75/20, kd=2），
# 但实测该配置下机器人连自带的 home / knees_bent 关键帧都站不住——
# 静态保持 1.34 秒就倒。物理系统本身站不住，奖励和超参怎么调都没用。
# 实测扫描：kp=250 + kd=20 静态站立 5 秒无碍，零动作前馈从 63 步提升到
# 154 步，关节跟踪误差 1.2°。kp 再高到 750 反而变差。
# 注意：真机 G1 的腿部 kp 约 100，此处偏高——先在仿真里跑通，
# sim2real 时需要配合域随机化把增益扫进训练分布。
KP_SCALE = 250.0             # 覆盖所有执行器的 kp
KD_RATIO = 1.4               # 阻尼比，1.0 为临界阻尼；略过阻尼以抑制接触振荡

# 腿部关节的阻尼下限。dampratio=1 按**关节自身惯量**算临界阻尼，这对
# 摆动腿和手臂是对的，但对支撑关节是错的：脚踩地时踝关节实际驱动的是
# 整个 33kg 倒立摆（等效惯量 m·h² ≈ 16 kg·m²，对应临界阻尼上百），
# 而不是那只脚（惯量 0.007）。v10 实测踝阻尼掉到 4.5 后：
#   MJX 存活 275.9 -> 231.4，CPU 各起点方差极大（63~500 步）
#   摔倒 20/23 次是渐进倾倒，倒地前 0.2 秒倾角已超 30° —— 看得见但救不回来
# 所以不该乘系数（腿内部惯量差 10 倍，一个系数会把髋顶到 188），
# 而该取下限：惯量值管肢体自身运动，下限管全身稳定，取两者较大。
KD_SUPPORT_MIN = 18.0        # v9 的 20.2 稳定性良好，取略低值兼顾灵活
SUPPORT_KEYS = ("hip", "knee", "ankle")

# 基准模型（scene.xml）用 <position kp="500" dampratio="1"/>，MuJoCo 会
# **按每个关节的等效惯量分别**算出 kd，范围 4.6~43.0，跨度近 10 倍。
# v9 之前我用了一个全局常数 kd=20.2 覆盖全部 29 个关节，后果是：
#   腕/踝 roll 过阻尼 6.3 倍、腕 yaw 5.9 倍 —— 手臂被摁住，动不起来
#   髋 pitch 反而只有 0.7 倍 —— 支撑腿欠阻尼
# 29 个关节里 18 个过阻尼超 2 倍，其中 12 个是手臂/腕部。实测表现为
# 肘腕跟踪误差最大（9~10.5°）而力矩最小（0.3~0.8 N·m），视觉上就是
# 「手臂拖着走」的僵硬感；机器人的高频能量占比(0.365%)甚至低于参考
# 动作(0.400%)——比参考还平滑，正是过阻尼的特征。
_BASE_XML = XML.parent/"scene.xml"


def _per_joint_kd(m):
    """per-joint kd：惯量算出的临界阻尼 × 比例，腿部再取下限。

    临界阻尼 kd = 2·√(kp·I)，I 是关节等效惯量。kp 变化时 kd 按平方根缩放
    才能保持同一阻尼比——这也是为什么不能用一个常数糊弄过去。
    """
    mb = mujoco.MjModel.from_xml_path(str(_BASE_XML))
    kd_base = -mb.actuator_biasprm[:m.nu, 2]      # dampratio=1 的临界阻尼
    kp_base = mb.actuator_gainprm[0, 0]
    kd = kd_base * np.sqrt(KP_SCALE / kp_base) * KD_RATIO

    # 腿部取下限，见 KD_SUPPORT_MIN 注释
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) or ""
        if any(k in nm for k in SUPPORT_KEYS):
            kd[i] = max(kd[i], KD_SUPPORT_MIN)
    return kd


def act_scale(m):
    """per-joint 残差幅度（rad）。见 ACT_SCALE_REF 注释。"""
    jid = m.actuator_trnid[:m.nu, 0]
    torque = np.abs(m.jnt_actfrcrange[jid]).max(axis=1)
    # √力矩：腕 5 N·m 与髋 139 N·m 差 28 倍，开方后降到 5.3 倍，
    # 既保留「强关节给更大修正范围」的意图，又不至于让弱关节几乎没权威
    rel = np.sqrt(torque)
    scale = ACT_SCALE_REF * rel / np.median(rel)
    half_range = (m.jnt_range[jid, 1] - m.jnt_range[jid, 0]) / 2
    return np.minimum(scale, ACT_RANGE_FRAC * half_range)


def configure_model(m, sim_dt=0.004):
    """训练和回放共用的模型配置。

    两边各写一份必然漂移——之前求解器设置就漏同步过一次，
    导致 CPU 评估测的其实是另一个物理系统。
    """
    m.opt.timestep = sim_dt
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.iterations = 4
    m.opt.ls_iterations = 8
    m.actuator_gainprm[:, 0] = KP_SCALE
    m.actuator_biasprm[:, 1] = -KP_SCALE
    m.actuator_biasprm[:, 2] = -_per_joint_kd(m)
    return m


# ---------------------------------------------------------------- 参考动作
def _slerp(q0, q1, t):
    """球面线性插值，q 为 wxyz，t 形状 (T,)。

    直接对四元数做线性插值会让转速在大角度时失真，且结果不再是
    单位四元数。动捕里髋/肩的单帧转角可以到十几度，必须用 slerp。
    """
    import numpy as np
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)            # 取最短弧
    d = np.clip(np.abs(d), -1.0, 1.0)
    th = np.arccos(d)
    st = np.sin(th)
    near = (d > 0.9995)                      # 近乎平行时 slerp 会除零，退化为 lerp
    st_safe = np.where(near, 1.0, st)
    tt = t[:, None]
    w0 = np.where(near, 1.0 - tt, np.sin((1.0 - tt) * th) / st_safe)
    w1 = np.where(near, tt,       np.sin(tt * th)         / st_safe)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _resample(qpos, src_fps, dst_fps):
    """把参考动作从 src_fps 重采样到 dst_fps。

    数据是 30fps，控制是 50Hz。不重采样的话每个控制步前进一个源帧，
    参考动作会被以 50/30 = 1.67 倍速播放，学出来的策略动作整体偏快。
    """
    import numpy as np
    T = len(qpos)
    T2 = int((T - 1) * dst_fps / src_fps) + 1
    t = np.arange(T2) * (src_fps / dst_fps)
    i0 = np.clip(np.floor(t).astype(np.int64), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    f = (t - i0)[:, None]
    pos = qpos[i0, :3] * (1 - f) + qpos[i1, :3] * f
    quat = _slerp(qpos[i0, 3:7], qpos[i1, 3:7], t - i0)
    jnt = qpos[i0, 7:] * (1 - f) + qpos[i1, 7:] * f
    return np.concatenate([pos, quat, jnt], axis=1)


def _finite_diff_qvel(qpos, dt):
    """从参考位姿差分出参考速度 (T, NV)。

    RSI 从动作中段起步，如果把 qvel 置零，等于让机器人以静止状态
    凭空出现在一个高速姿态里——第一步就会因为动量不匹配而崩掉。
    MuJoCo 自由关节约定：qvel[0:3] 是世界系线速度，
    qvel[3:6] 是**机体系**角速度。
    """
    import numpy as np
    T = len(qpos)
    v = np.zeros((T, NV))
    d = np.diff(qpos, axis=0)                       # (T-1, NQ)
    v[:-1, 0:3] = d[:, :3] / dt                     # 线速度，世界系
    v[:-1, 6:] = d[:, 7:] / dt                      # 关节速度

    q0, q1 = qpos[:-1, 3:7], qpos[1:, 3:7]
    # dq = conj(q0) ⊗ q1，即 q0 到 q1 的相对旋转（机体系）
    w0, x0, y0, z0 = q0[:, 0], -q0[:, 1], -q0[:, 2], -q0[:, 3]
    w1, x1, y1, z1 = q1[:, 0],  q1[:, 1],  q1[:, 2],  q1[:, 3]
    dw = w0*w1 - x0*x1 - y0*y1 - z0*z1
    dx = w0*x1 + x0*w1 + y0*z1 - z0*y1
    dy = w0*y1 - x0*z1 + y0*w1 + z0*x1
    dz = w0*z1 + x0*y1 - y0*x1 + z0*w1
    s = np.sign(dw)                                  # 取最短弧，避免 ±2π 跳变
    s[s == 0] = 1.0
    v[:-1, 3:6] = 2.0 * np.stack([dx, dy, dz], axis=1) * s[:, None] / dt

    v[-1] = v[-2] if T > 1 else 0.0
    return v


def _body_states(qpos_seq, sim_dt=0.004):
    """对参考轨迹逐帧做 FK：跟踪 body 的世界系位置 + 触地 geom 计数。

    运行时在 MJX 里做 FK 太贵，且参考是固定的——一次算好存成数组。
    返回 (身体位置 (T, 1+B, 3), 接触点数 (T,))。第 0 个 body 是 anchor。
    接触计数直接用 MuJoCo 碰撞检测的 d.ncon（mj_forward 已含碰撞阶段），
    不要自算「表面低于阈值的 geom 数」——12060 那帧有 2 个 geom 贴地但
    都在同一只脚上，geom 计数漏掉单点支撑，第一版就栽在这。
    """
    m = configure_model(mujoco.MjModel.from_xml_path(str(XML)), sim_dt)
    d = mujoco.MjData(m)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in (ANCHOR_BODY,) + TRACK_BODIES]
    assert all(i >= 0 for i in ids), "TRACK_BODIES 里有名字对不上模型"

    out = np.zeros((len(qpos_seq), len(ids), 3))
    ncon = np.zeros(len(qpos_seq), dtype=np.int8)
    for t, q in enumerate(qpos_seq):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)          # 只需运动学，不积分
        out[t] = d.xpos[ids]
        ncon[t] = min(d.ncon, 127)
    return out, ncon


def load_reference(names, ctrl_dt=0.02, max_frames=None):
    """把若干段动作拼成参考库。

    返回 (位姿 (N,T,NQ), 速度 (N,T,NV), 身体位置 (N,T,1+B,3),
          接触数 (N,T), 各段真实长度 (N,))。

    **补齐而不是裁齐。** jit 要求规则数组，早期实现裁到 min(长度)——
    68 段数据里最短的 fallAndGetUp3 只有 5109 帧，会把 15286 帧的长段
    一起砍掉，浪费 52.4% 的数据。改为补齐到 max：末帧重复填充（不是补零，
    补零会让参考跳到原点，早停立刻触发），另外返回真实长度，
    RSI 采样和参考索引都按各段自己的长度来。
    显存代价可接受：68 段补齐到 15286 帧约 420MB。
    """
    import numpy as np
    dst_fps = 1.0 / ctrl_dt
    poses, vels, keep = [], [], []
    for n in names:
        p = DATASET/f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        q = _resample(z["qpos"], float(z["fps"]), dst_fps)
        if max_frames:
            q = q[:max_frames]
        poses.append(q)
        vels.append(_finite_diff_qvel(q, ctrl_dt))
        keep.append(n)
    if not poses:
        raise SystemExit(f"没找到任何动作，检查 {DATASET}")

    lens = np.array([len(c) for c in poses], dtype=np.int32)
    T = int(lens.max())

    def pad(arr):
        """末帧重复补齐到 T。"""
        if len(arr) >= T:
            return arr[:T]
        tail = np.repeat(arr[-1:], T - len(arr), axis=0)
        return np.concatenate([arr, tail], axis=0)

    fk = [_body_states(pad(c)) for c in poses]
    return (jp.asarray(np.stack([pad(c) for c in poses]), dtype=jp.float32),
            jp.asarray(np.stack([pad(c) for c in vels]), dtype=jp.float32),
            jp.asarray(np.stack([b for b, _ in fk]), dtype=jp.float32),
            np.stack([n for _, n in fk]),        # (N, T) 接触数，numpy 即可
            lens)                                 # (N,) 各段真实长度


class G1Imitate(PipelineEnv):
    """单段/多段动作模仿。"""

    def __init__(self, ref_qpos, ref_qvel, ref_body, ref_ncon=None,
                 ref_lens=None, ctrl_dt=0.02, sim_dt=0.004, ep_len=500,
                 **kwargs):
        mj_model = configure_model(
            mujoco.MjModel.from_xml_path(str(XML)), sim_dt)

        sys = mjcf.load_model(mj_model)
        n_frames = int(round(ctrl_dt / sim_dt))
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames, **kwargs)

        self._ref = ref_qpos                    # (N, T, NQ)，已重采样到控制率
        self._refv = ref_qvel                   # (N, T, NV)
        self._refb = ref_body                   # (N, T, 1+B, 3) 世界系身体位置
        # 机器人侧对应的 body id（顺序与 _body_states 一致：anchor 在前）
        self._body_ids = jp.asarray(
            [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
             for n in (ANCHOR_BODY,) + TRACK_BODIES])
        self._n_clip, self._T = ref_qpos.shape[0], ref_qpos.shape[1]
        self._mj_model = mj_model

        # 参考已经在控制率上，一个控制步正好前进一帧
        self._ref_stride = 1

        # 各段的真实长度（补齐前）。参考索引必须按各自长度钳制，
        # 否则短段会一直读到补齐区的重复末帧。
        if ref_lens is None:
            ref_lens = np.full(self._n_clip, self._T, dtype=np.int32)
        self._lens = jp.asarray(np.asarray(ref_lens), dtype=jp.int32)

        # RSI 起点上界：留出整个回合的余量，这样回合内永远不会播完参考。
        # 否则「参考播完」会和「摔倒」共用 done，价值函数会把片段结束
        # 误学成失败状态。多段时每段的上界不同。
        max_start = np.maximum(1, np.asarray(ref_lens) - ep_len - LOOKAHEAD - 2)
        self._max_start_arr = jp.asarray(max_start, dtype=jp.int32)
        self._max_start = int(max_start.min())     # 兼容旧代码的标量引用

        # 有效起点表：排除接触点不足的帧（见 MIN_START_CONTACTS 注释）。
        # 各段的有效帧数不同，无法直接做规则数组——把每段的有效索引
        # 循环平铺到统一长度 K，reset 时均匀抽列号即可（平铺余数带来的
        # 采样偏差 < 1/K，可忽略）。
        if ref_ncon is not None:
            valid = []
            for c in range(self._n_clip):
                lim = int(max_start[c])
                ok = np.where(np.asarray(ref_ncon[c][:lim])
                              >= MIN_START_CONTACTS)[0]
                if len(ok) == 0:                   # 兜底：不过滤
                    ok = np.arange(lim)
                valid.append(ok)
            K = max(len(v) for v in valid)
            table = np.stack([np.resize(v, K) for v in valid])
            self._valid_starts = jp.asarray(table, dtype=jp.int32)   # (N, K)
            self._n_valid = K
        else:                                      # 兼容旧调用：均匀采样
            self._valid_starts = None
            self._n_valid = self._max_start

        # 关节限位，用于把残差目标钳制在合法范围内
        self._jnt_lo = jp.asarray(mj_model.jnt_range[1:, 0], dtype=jp.float32)
        self._jnt_hi = jp.asarray(mj_model.jnt_range[1:, 1], dtype=jp.float32)
        self._act_scale = jp.asarray(act_scale(mj_model), dtype=jp.float32)

    # ------------------------------------------------------------ 工具
    def _phase(self, info):
        """当前参考帧索引。

        **必须**从 info["steps"] 推导，不能自己维护累加计数器。
        brax 的 AutoResetWrapper 在回合结束时把 pipeline_state 恢复成
        起始位姿、把 info["steps"] 清零，但它只 where_done 了
        first_pipeline_state / first_obs 两项，**不会**恢复环境自己写进
        info 的字段。自维护的计数器于是只增不减，从第二个回合起参考帧
        就和机器人的实际位姿永久脱钩，策略等于在拟合噪声。
        info["start"] 在 reset 里写一次、之后永不修改，所以能安全存活。
        """
        if "steps" in info:                       # 被 EpisodeWrapper 包过
            return info["start"] + info["steps"].astype(jp.int32)
        return info["step"]                       # 裸环境（自测/回放）

    def _last(self, clip):
        """该段的最后一个有效帧下标（补齐区不算）。"""
        return self._lens[clip] - 1

    def _ref_at(self, clip, step):
        """取参考帧，超出该段末尾则钳制（按各段真实长度，不是补齐后的 T）。"""
        idx = jp.clip(step, 0, self._last(clip))
        return self._ref[clip, idx], self._refv[clip, idx]

    def _ref_future(self, clip, step):
        """一次取出未来 LOOKAHEAD 帧的关节角。"""
        idx = jp.clip(step + jp.arange(1, LOOKAHEAD + 1), 0, self._last(clip))
        return self._ref[clip, idx, 7:].reshape(-1)

    @staticmethod
    def _rot_t(quat):
        """机体<-世界的旋转矩阵 R(q)^T。"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        r = jp.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])
        return r.T

    def _catch_up(self, ref, refv, qpos):
        """带位置回正的目标速度 + 位置误差，均为世界系。见 K_CATCH 注释。"""
        perr = ref[:3] - qpos[:3]
        catch = jp.clip(K_CATCH * perr, -V_CATCH_MAX, V_CATCH_MAX)
        return refv[0:3] + catch, perr

    def _obs(self, data, clip, step, last_act):
        qpos, qvel = data.qpos, data.qvel
        # 重力在机身系的投影（真机可从 IMU 得到）
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        grav = jp.array([-2*(x*z - w*y), -2*(y*z + w*x), -(1 - 2*(x*x + y*y))])
        # 目标速度和位置误差都放机身系：不引入全局朝向依赖，也贴合真机
        # （base velocity 由 IMU + 腿部里程计估计，本来就在机身系）
        ref, refv = self._ref_at(clip, step)
        vt, perr = self._catch_up(ref, refv, qpos)
        rt = self._rot_t(qpos[3:7])
        return jp.concatenate([grav, qvel[3:6], qpos[7:], qvel[6:], last_act,
                               rt @ vt, rt @ perr,
                               self._ref_future(clip, step)])

    # ------------------------------------------------------------ 接口
    def reset(self, rng: jax.Array) -> State:
        rng, k1, k2 = jax.random.split(rng, 3)
        clip = jax.random.randint(k1, (), 0, self._n_clip)
        # 随机起始相位（RSI, reference state initialization）——
        # DeepMimic 的关键技巧：不从头开始，否则后半段永远学不到。
        # 只从有效起点表里抽：单点支撑的帧不适合静态初始化（见表注释）
        if self._valid_starts is not None:
            j = jax.random.randint(k2, (), 0, self._n_valid)
            step = self._valid_starts[clip, j]
        else:
            step = jax.random.randint(k2, (), 0, self._max_start)

        qpos, qvel = self._ref_at(clip, step)     # 位姿和速度都取自参考
        data = self.pipeline_init(qpos, qvel)

        last_act = jp.zeros(NU)
        obs = self._obs(data, clip, step, last_act)
        # start 只在这里写一次，之后永不修改——它必须在 AutoResetWrapper
        # 的重置中存活下来，见 _phase()
        info = {"clip": clip, "start": step, "step": step,
                "last_act": last_act, "rng": rng}
        metrics = {"r_body": 0.0, "r_pose": 0.0, "r_orient": 0.0,
                   "r_root": 0.0, "r_rvel": 0.0, "r_jvel": 0.0,
                   "r_alive": 0.0, "r_effort": 0.0, "body_err": 0.0,
                   "pose_err": 0.0, "root_err": 0.0, "rvel_err": 0.0}
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        clip = state.info["clip"]
        nstep = self._phase(state.info) + 1
        ref, refv = self._ref_at(clip, nstep)

        # 残差动作：目标 = 下一参考帧关节角 + 有界修正量（幅度按关节而异）
        ctrl = jp.clip(ref[7:] + self._act_scale * jp.clip(action, -1.0, 1.0),
                       self._jnt_lo, self._jnt_hi)
        data = self.pipeline_step(state.pipeline_state, ctrl)

        # --- 跟踪误差 ---
        # 主项改为**笛卡尔身体位置**（相对 anchor），不再用关节角。
        # 关节角误差不区分杠杆，而空间位置天然按运动学杠杆加权。
        # 相对 anchor 是为了把「机器人在哪」（下面的 root_err）和
        # 「摆成什么形状」解耦——两者混在一起时会互相牵制。
        rb = data.xpos[self._body_ids]                 # (1+B, 3) 机器人
        tb = self._refb[clip, jp.clip(nstep, 0, self._last(clip))]   # 参考
        rel_r = rb[1:] - rb[0]
        rel_t = tb[1:] - tb[0]
        body_err = jp.mean(jp.sum(jp.square(rel_r - rel_t), axis=-1))

        # 关节角项保留但降权，作为姿态的弱正则（防止出现空间位置对、
        # 但关节配置离谱的解，例如镜像或多解 IK）
        pose_err = jp.mean(jp.square(data.qpos[7:] - ref[7:]))
        root_err = jp.sum(jp.square(data.qpos[:3] - ref[:3]))
        quat_err = 1.0 - jp.abs(jp.dot(data.qpos[3:7], ref[3:7]))

        # 速度跟踪的目标带位置回正项，跟踪它就等价于把位置误差拉回零
        vt, _ = self._catch_up(ref, refv, data.qpos)
        rvel_err = jp.sum(jp.square(data.qvel[0:3] - vt))
        jvel_err = jp.mean(jp.square(data.qvel[6:] - refv[6:]))

        r_body = jp.exp(-K_BODY * body_err)
        r_pose = jp.exp(-K_POSE * pose_err)
        r_root = jp.exp(-K_ROOT * root_err)
        r_orient = jp.exp(-K_ORIENT * quat_err)
        r_rvel = jp.exp(-K_RVEL * rvel_err)
        r_jvel = jp.exp(-K_JVEL * jvel_err)

        upright = 1.0 - 2.0 * (data.qpos[4]**2 + data.qpos[5]**2)
        fell = (data.qpos[2] < 0.2) | (upright < 0.0)
        # DeepMimic 的关键技巧：跟丢了就终止回合。不加这条的话，
        # 「无视参考、站着不摔」是个稳定的局部最优——它能稳拿存活分，
        # 而跟踪分反正也快拿不到。上一轮训练正是掉进了这里。
        # 早停也改用笛卡尔误差为主：关节角阈值放宽为兜底，
        # 因为深蹲这类姿态下关节角误差不能准确反映「跟丢了没有」
        lost = ((body_err > MAX_BODY_ERR)
                | (pose_err > MAX_POSE_ERR)
                | (root_err > MAX_ROOT_ERR))
        bad = fell | lost
        r_alive = jp.where(bad, 0.0, 1.0)

        r_effort = -0.001 * jp.sum(jp.square(action))

        reward = (W_BODY*r_body + W_POSE*r_pose + W_ROOT*r_root
                  + W_ORIENT*r_orient + W_RVEL*r_rvel + W_JVEL*r_jvel
                  + W_ALIVE*r_alive + r_effort)

        # done = 摔倒或跟丢。参考播完不会在回合内发生（见 _max_start），
        # 回合到点属于截断，由 brax 的 EpisodeWrapper 写 truncation。
        done = jp.where(bad, 1.0, 0.0)

        obs = self._obs(data, clip, nstep, action)
        state.info["step"] = nstep
        state.info["last_act"] = action
        state.metrics.update(r_body=r_body, r_pose=r_pose, r_orient=r_orient,
                             r_root=r_root, r_rvel=r_rvel, r_jvel=r_jvel,
                             r_alive=r_alive, r_effort=r_effort,
                             body_err=body_err, pose_err=pose_err,
                             root_err=root_err, rvel_err=rvel_err)
        return state.replace(pipeline_state=data, obs=obs,
                             reward=reward, done=done)
