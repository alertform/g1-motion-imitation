#!/usr/bin/env bash
A="$HOME/tools/rl/runs/multi_v16_noclipcond/train.log"
B="$HOME/tools/rl/runs/multi/train.log"
echo "=== 同步数对比（训练指标：回合长）==="
printf "  %-12s %10s %10s %8s\n" "步数" "v16 无条件" "v17 有条件" "变化"
for s in 40960000 81920000 122880000 163840000 204800000 245760000 \
         286720000 327680000 368640000 399360000; do
    a=$(grep "^ *$s 步" "$A" 2>/dev/null | awk '{print $6}')
    b=$(grep "^ *$s 步" "$B" 2>/dev/null | awk '{print $6}')
    if [ -n "$a" ] && [ -n "$b" ]; then
        d=$(echo "$a $b" | awk '{printf "%+.0f%%", ($2/$1-1)*100}')
    else
        d="-"
    fi
    printf "  %-12s %10s %10s %8s\n" "$s" "${a:--}" "${b:--}" "$d"
done
echo
echo "=== v17 全程分段中枢 ==="
grep '步   奖励' "$B" | awk '{s+=$6; n++; if(n%20==0){printf "  至 %6.0fM 步: %6.1f\n", $1/1000000, s/20; s=0}}'
echo
echo "=== 各自最终值 ==="
echo "  v16 末尾: $(tail -5 "$A" | grep '步   奖励' | tail -1)"
echo "  v17 末尾: $(tail -5 "$B" | grep '步   奖励' | tail -1)"
