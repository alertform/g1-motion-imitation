#!/usr/bin/env bash
# 日志行格式：  <步数> 步   奖励  <值>   回合长  <值>   <分钟> 分钟
#              $1      $2   $3    $4     $5     $6      $7    $8
L="$HOME/tools/rl/runs/walk/train.log"
echo "=== 最近 4 条（日志实值）==="
tail -4 "$L"
echo
echo "=== 分段中枢：每 20 个评估点（约 41M 步）的回合长均值 ==="
grep '步   奖励' "$L" | awk '{s+=$6; n++; if(n%20==0){printf "  至 %6.0fM 步: 均值 %6.1f\n", $1/1000000, s/20; s=0}}'
echo
echo "=== 最近 20 点 vs 上一个 20 点 ==="
grep '步   奖励' "$L" | tail -40 | awk 'NR<=20{a+=$6}NR>20{b+=$6}END{if(a>0)printf "  上一段 %.1f -> 最近段 %.1f  (%+.1f%%)\n", a/20, b/20, (b/a-1)*100}'
