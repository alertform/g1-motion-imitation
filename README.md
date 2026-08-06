# G1 动作模仿

把人类动捕数据重定向到 Unitree G1，再用强化学习在物理仿真中学会实际执行。

```
LAFAN1 动捕 (BVH)
   ↓  GMR 差分 IK 重定向          src/retarget/
G1 关节轨迹（运动学，未必物理可行）
   ↓  后处理 + 审计筛选            src/postprocess/
68 段可用参考动作（438k 帧）
   ↓  DeepMimic 式 RL（MJX + PPO） src/rl/
可在物理仿真中执行的控制策略
```

## 当前状态

**v11 完成**，单段行走 `walk1_subject1`。MJX 下 16 个起点 × 1500 步评估：

| | 零动作前馈 | v11 |
|---|---|---|
| 存活均值 | 56.5 | **1492.6** |
| 存活中位 | 49.5 | **1500**（满） |
| 策略/前馈 | 1.00 | **26.42** |
| 根漂移 | 22.41cm | **4.32cm** |
| 根速度误差 | 1.248 m/s | **0.197 m/s** |
| 关节误差 | 3.55° | 5.84° |

**15/16 个起点跑满 1500 步（30 秒不摔）**，最后一个 1382 步。
完整实验记录见 [docs/experiments.md](docs/experiments.md)。

下一步方向（瓶颈已从「训练量」转为「分布覆盖」与「架构」）：

- **自适应采样** / **多段数据** —— 68 段数据尚未使用
- **移植 BeyondMimic 的设计**：笛卡尔身体跟踪奖励 + anchor 机制、
  真实电机惯量、推导式 PD 增益、per-joint 动作缩放、动作率惩罚

## 环境

| | |
|---|---|
| 平台 | WSL2 + Ubuntu 24.04（Windows 11 宿主）|
| 仿真 | MuJoCo 3.11 + MJX |
| RL | Brax PPO，JAX CUDA |
| GPU | RTX 4060 8GB，4096 并行环境，约 12,000 env-步/秒 |
| 机器人 | `mujoco_menagerie/unitree_g1`（29 自由度）|
| 数据 | LAFAN1（Ubisoft，CC BY-NC-ND 4.0）|

## 目录

```
src/rl/            RL 环境、训练、评估、回放
src/retarget/      GMR 之上的重定向（接触约束、标定）
src/postprocess/   去滑步、贴地、平滑、审计
src/motions.py     动作浏览器（viewer 里逐帧看数据）
tutorial/          MuJoCo 十课入门
scripts/           运行与诊断脚本
docs/              实验记录、决策与未解问题
papers/            参考文献索引（PDF 不入库，见 scripts/get-papers*.sh）
```

## 常用命令

脚本都在 WSL 里跑，路径 `/mnt/d/g1-imitation/`。

```bash
# 训练（脱离式，参数为步数；第二个参数可选，用于续训）
bash scripts/train-walk-detach.sh 800000000
bash scripts/train-walk-detach.sh 500000000 walk_v11_final/policy.pkl

# 评估 —— 判断训练是否有效必须用 MJX 版（与训练同引擎）
bash scripts/run-eval-mjx.sh --episodes 16 --max-steps 1500
bash scripts/run-eval.sh      # CPU 版，更接近真机，用于估计 sim2real 差距

# viewer 回放
bash scripts/play-v7.sh --speed 0.5

# 诊断
bash scripts/diag-stiff.sh    # 僵硬来源：增益 / 抖动 / 力矩饱和
bash scripts/diag-fall.sh     # 摔倒模式：随机失稳还是参考不可行
bash scripts/show-params.sh   # 列出全部可调参数与当前值
bash scripts/v8-trend.sh      # 训练曲线分段中枢（单点噪声大，必须看均值）
```

## 四条踩过坑的准则

**1. 奖励曲线上升不代表在学对的东西**

v1 的奖励从 6.3 涨到 72.5，客观指标却是关节误差 38.7°、根漂移 63cm —— 完全没在模仿。
必须同时看三个量：奖励、**每步回报**、**折算成度和厘米的客观误差**。

**2. 评估口径必须与训练一致**

MJX 与 CPU MuJoCo 物理不等价（同一段前馈，某起点 CPU 371 步 vs MJX 53 步）。
判断训练是否有效用 `run-eval-mjx.sh`，CPU 版只用于估计 sim2real 差距。

**3. 同一个量不要算两遍**

观测维度、求解器设置、增益配置都曾因为「训练和评估各写一份」而漂移，
测出来的是另一个物理系统。现在统一走 `rl_env.configure_model()` 和 `rl_env.OBS_SIZE`。

**4. 机制无效时，先查执行链路再改设计**

根漂移在 v8/v9/v10 卡在 25cm，我判断是奖励权重的平衡点、要改权重。
实际是踝关节阻尼不足导致控制权威不够 —— v9 引入的位置回正机制一直是对的，
只是执行机构使不上劲。补上阻尼后漂移直接降到 4.32cm。

## 数据来源

LAFAN1 采用 CC BY-NC-ND 4.0 许可（Ubisoft），仅限非商业使用。数据集与重定向结果不在本仓库中。
