# 参考文献索引

对应 G1 动作模仿项目（代码 `D:\WSL\rl_*.py`，决策记录 `D:\WSL\UNKNOWNS.md`）。
每篇都标注了它对应我们做过的哪个决策，或还没用上但值得知道。

```
01-核心方法/          直接决定了当前实现，必读
02-动作重定向/        人 -> 机器人的动作映射
03-训练方法/          超参、规模律、替代技术路线
04-多动作与自然度/    68 段数据怎么用、动作怎么才像人
05-sim2real/          上真机需要的
```

---

## 01-核心方法

### DeepMimic_Peng2018.pdf
*DeepMimic: Example-Guided Deep RL of Physics-Based Character Skills* (SIGGRAPH 2018)

**我们整套骨架的来源。** 三个核心机制全部沿用：

| 论文机制 | 我们的实现 | 对应教训 |
|---|---|---|
| RSI（参考状态初始化）| `G1Imitate.reset()` 随机起始相位 | 不加的话动作后半段永远学不到 |
| Early Termination | `MAX_POSE_ERR` / `MAX_ROOT_ERR` | **v1 失败的直接原因**：没有早停，「无视参考站着不动」是稳定局部最优 |
| `r = Σ ωᵢ·exp(-kᵢ·errᵢ)` | 6 项加权奖励 | **v1 的第二个 bug**：29 个关节误差求和进指数，20° 就饱和，改均值才有梯度 |

论文原话值得记：RSI 相当于一个「顾问」，告诉角色哪些状态在正确执行时能拿高回报。

### RealWorldHumanoidLocomotion_ScienceRobotics.pdf
*Real-World Humanoid Locomotion with Reinforcement Learning*

**动作空间设计的依据**，也是**当前困境的可能出路**。

- 策略输出关节目标角、PD 转力矩 —— 我们 v3 的残差动作就是这个范式
- 典型增益 kp=200 / kd=10，和我们摸索出的 kp=250 同量级
- **论文把 PD 增益本身也放进动作空间**（预测 8 个腿部关节的增益），理由是「接触高度不确定时，让增益随状态变化能提升性能」

最后一点直接命中我们的问题：v9 常数 kd 太僵、v10 per-joint 踝部欠阻尼 —— **可能根本不存在一组同时满足支撑期和摆动期的固定值**。

### LeggedLocomotion_Survey.pdf
*Learning-based Legged Locomotion: State of the Art and Future Perspectives*

综述，建立全局视野用。域随机化、sim2real、动作空间选择都有系统梳理。

---

## 02-动作重定向

### ReActor_PhysicsAwareRetargeting.pdf
*RL for Physics-Aware Motion Retargeting* (2026)

**和我们流程高度相关。** 我们是两段式：纯运动学重定向（GMR）→ RL 补物理可行性。它把物理直接放进重定向环节。若奏效，能省掉现在这一整轮 RL 的很多麻烦。

### KinematicRetargeting_ContactRich.pdf
接触丰富场景的重定向。我们审计发现的脚部穿透 7%、脚倾角偏差属同类问题。

### SoftRobotRetargeting.pdf
*Functional Force-Aware Retargeting to Soft Robot Policies* (2026)

核心洞察：传统方法依赖人机**运动学对应**（映射关节角），对非拟人或软体结构显著失效，须改用任务空间 / 接触 / 力的对齐。

---

## 03-训练方法

### EmbodimentScalingLaws.pdf
*Towards Embodiment Scaling Laws in Robot Locomotion*

**PPO 超参的依据。** 对照检查我们的配置：

| 参数 | 文献 | 我们 | 状态 |
|---|---|---|---|
| 学习率 | ~3e-4 | 3e-4 | ✓ |
| target KL | 0.01~0.02 | 0.01 | ✓ |
| clip | ~0.2 | ~~0.3~~ → 0.2 | **v11 已修** |
| 熵系数 | 0~0.01 | 1e-3 | ✓ |
| 更新轮数 | 3-5 | 4 | ✓ |

`clipping_epsilon` 原本用的是 brax 默认 0.3，偏大。clip 越大单次更新允许的策略变化越大 —— 与 v2 观察到的「更新破坏平衡能力」可能有关。

### DiffMimic.pdf
可微物理做动作模仿，DeepMimic（无模型 RL）的替代路线。样本效率高得多，但依赖物理可微、易陷局部最优。**我们没走这条路，但值得知道存在。**

### mjlab.pdf
GPU 加速机器人学习框架。我们是 MJX + Brax 手搭环境，可对照别人的工程选择。

---

## 04-多动作与自然度

### PHC_PerpetualHumanoidControl.pdf
*Perpetual Humanoid Control for Real-time Simulated Avatars*

**「一个策略吃多段动作」的标杆**：单策略模仿整个 AMASS 数据集，99% 成功率。我们的 68 段数据要用起来，这是主要参考。

还有一点：它用 AMP 处理**失败恢复**（摔了怎么自然爬起来），而我们现在摔了直接终止回合。

### AMP_AdversarialMotionPriors.pdf
*Adversarial Motion Priors for Stylized Physics-Based Character Control*

**思路和我们完全不同**：不做逐帧跟踪，用判别器保证「看起来像那类动作」。我们是硬跟踪参考。如果「像不像人」始终解决不好，这是另一条路。

### MultiDomainMotionEmbedding.pdf
把动作编码成嵌入向量、策略以嵌入为条件。解决「观测里怎么表示当前该做哪段动作」这个具体问题。

### MultipleAMP_AdvancedSkills.pdf
多个对抗先验组合学技能。

### RuN_ResidualPolicyNaturalLocomotion.pdf
残差策略做自然人形运动。我们的残差动作空间（`ctrl = 参考帧 + ACT_SCALE·act`）是同类思想。

---

## 05-sim2real

### DomainRandomization_SystematicSim2Real.pdf
*Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots*

**该随机化哪些参数的清单**：

| 类别 | 参数 |
|---|---|
| 动力学 | 各连杆质量、质心位置、电机功率 |
| 接触 | 地面摩擦、地面刚度 |
| 控制 | **PD 增益**、电机摩擦、**控制延迟** |
| 感知 | 观测噪声 |
| 外部 | 负载、躯干外力扰动 |

注意 **PD 增益本身就在清单里** —— 我们纠结「kp=250 偏离真机的 100」，标准做法不是找准值，而是把它扫进训练分布。

### OfflineDomainRandomization_Provable.pdf
离线域随机化的理论保证。

### SoftBodyFeet_DigitalTwin.pdf
刚体骨架 + 软体脚底。若后续发现脚地接触是 sim2real 差距主因，比调 `solref`/`solimp` 更物理。代价是仿真变慢。

---

## 还没找但值得找

- **失败恢复**：摔倒后爬起来（PHC 用 AMP 做，我们直接终止）
- **课程学习**：从简单动作逐步过渡到复杂动作
- **接触参数辨识**：`solref`/`solimp` 怎么对齐真机
