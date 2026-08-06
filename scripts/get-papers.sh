#!/usr/bin/env bash
# 下载本项目引用过的论文。放到 /mnt/d/WSL/papers，Windows 侧可直接打开。
DIR=/mnt/d/WSL/papers
mkdir -p "$DIR"

# 格式: arXiv编号|文件名|一句话说明
PAPERS='
1804.02717|DeepMimic_Peng2018|奖励设计+RSI+早停，我们整套骨架的来源
2303.03381|RealWorldHumanoidLocomotion_ScienceRobotics|关节位置动作空间，PD增益也进动作空间
2406.01152|LeggedLocomotion_Survey|腿式运动学习综述，看全局用
2304.03274|DiffMimic|可微物理做动作模仿，DeepMimic的替代路线
2402.04820|KinematicRetargeting_ContactRich|接触丰富场景的运动学重定向
2411.14701|SoftBodyFeet_DigitalTwin|软体脚底的行走仿真
2605.06593|ReActor_PhysicsAwareRetargeting|物理感知的重定向（RL做重定向）
2604.01224|SoftRobotRetargeting|人类演示到软体机器人的重定向
2601.22074|mjlab|GPU加速机器人学习框架
'

echo "=== 开始下载到 $DIR ==="
ok=0; fail=0
while IFS='|' read -r id name desc; do
    [ -z "$id" ] && continue
    out="$DIR/${name}.pdf"
    if [ -s "$out" ]; then
        echo "  已存在  $name.pdf"
        ok=$((ok+1)); continue
    fi
    curl -sL --max-time 90 -o "$out" "https://arxiv.org/pdf/${id}" 2>/dev/null
    # 校验：必须是 PDF 且有合理大小，别把 404 页面当成功
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$sz" -gt 50000 ] && head -c 4 "$out" | grep -q '%PDF'; then
        printf "  OK      %-45s %6.1f MB\n" "$name.pdf" "$(echo "$sz/1048576" | bc -l)"
        ok=$((ok+1))
    else
        echo "  失败    $name  (arXiv:$id, 大小 $sz)"
        rm -f "$out"; fail=$((fail+1))
    fi
done <<< "$PAPERS"

echo
echo "=== 成功 $ok  失败 $fail ==="
ls -la "$DIR"/*.pdf 2>/dev/null | awk '{printf "  %8.1f KB  %s\n", $5/1024, $9}'
