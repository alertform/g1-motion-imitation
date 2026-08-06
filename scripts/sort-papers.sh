#!/usr/bin/env bash
DIR=/mnt/d/WSL/papers
cd "$DIR" || exit 1

declare -A CAT=(
  # 01 核心方法：直接决定了当前实现
  [DeepMimic_Peng2018]="01-核心方法"
  [RealWorldHumanoidLocomotion_ScienceRobotics]="01-核心方法"
  [LeggedLocomotion_Survey]="01-核心方法"
  # 02 动作重定向
  [ReActor_PhysicsAwareRetargeting]="02-动作重定向"
  [KinematicRetargeting_ContactRich]="02-动作重定向"
  [SoftRobotRetargeting]="02-动作重定向"
  # 03 训练方法：超参、规模律、替代路线
  [EmbodimentScalingLaws]="03-训练方法"
  [DiffMimic]="03-训练方法"
  [mjlab]="03-训练方法"
  # 04 多动作与自然度
  [PHC_PerpetualHumanoidControl]="04-多动作与自然度"
  [AMP_AdversarialMotionPriors]="04-多动作与自然度"
  [MultiDomainMotionEmbedding]="04-多动作与自然度"
  [MultipleAMP_AdvancedSkills]="04-多动作与自然度"
  [RuN_ResidualPolicyNaturalLocomotion]="04-多动作与自然度"
  # 05 sim2real
  [DomainRandomization_SystematicSim2Real]="05-sim2real"
  [OfflineDomainRandomization_Provable]="05-sim2real"
  [SoftBodyFeet_DigitalTwin]="05-sim2real"
)

for d in 01-核心方法 02-动作重定向 03-训练方法 04-多动作与自然度 05-sim2real; do
    mkdir -p "$d"
done

moved=0; miss=0
for f in *.pdf; do
    [ -e "$f" ] || continue
    base="${f%.pdf}"
    tgt="${CAT[$base]}"
    if [ -n "$tgt" ]; then
        mv "$f" "$tgt/"; moved=$((moved+1))
    else
        echo "  未分类: $f"; miss=$((miss+1))
    fi
done

echo "=== 已归类 $moved 篇，未分类 $miss 篇 ==="
echo
for d in 0*/; do
    echo "$d"
    ls "$d"*.pdf 2>/dev/null | sed "s|$d||; s|^|    |"
done
