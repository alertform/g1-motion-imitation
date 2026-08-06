#!/usr/bin/env bash
DIR=/mnt/d/WSL/papers
mkdir -p "$DIR"

PAPERS='
2509.06342|DomainRandomization_SystematicSim2Real|系统性 sim2real，该随机化哪些参数
2506.10133|OfflineDomainRandomization_Provable|离线域随机化的理论保证
2505.05753|EmbodimentScalingLaws|跨形态的规模律，含 PPO 超参依据
2305.06456|PHC_PerpetualHumanoidControl|单策略模仿 AMASS 全库，99% 成功率
2104.02180|AMP_AdversarialMotionPriors|对抗式动作先验，自然度的另一条路
2512.07673|MultiDomainMotionEmbedding|多动作嵌入，一个策略吃多种动作
2509.20696|RuN_ResidualPolicyNaturalLocomotion|残差策略做自然人形运动
2203.14912|MultipleAMP_AdvancedSkills|多个对抗先验组合学技能
'

echo "=== 下载到 $DIR ==="
ok=0; fail=0
while IFS='|' read -r id name desc; do
    [ -z "$id" ] && continue
    out="$DIR/${name}.pdf"
    if [ -s "$out" ]; then echo "  已存在  $name.pdf"; ok=$((ok+1)); continue; fi
    curl -sL --max-time 90 -o "$out" "https://arxiv.org/pdf/${id}" 2>/dev/null
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$sz" -gt 50000 ] && head -c 4 "$out" | grep -q '%PDF'; then
        printf "  OK      %-42s %6.1f MB\n" "$name.pdf" "$(echo "$sz/1048576" | bc -l)"
        ok=$((ok+1))
    else
        echo "  失败    $name  (arXiv:$id, 大小 $sz)"; rm -f "$out"; fail=$((fail+1))
    fi
done <<< "$PAPERS"

echo
echo "=== 成功 $ok  失败 $fail   目录共 $(ls "$DIR"/*.pdf 2>/dev/null | wc -l) 篇 ==="
