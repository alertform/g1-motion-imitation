#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import'
"""68 段数据的实际形态——决定多段方案怎么实现。

关键问题：现在的 load_reference 把所有段裁到**最短那段的长度**
（T = min），多段时会浪费大量数据。先量清楚。
"""
import json, pathlib, numpy as np
from collections import Counter

D = pathlib.Path.home()/"tools"/"g1_dataset"
mf = json.loads((D/"manifest.json").read_text())["motions"]
usable = [r for r in mf if r["grade"].startswith("可用")]
print(f"manifest: {len(mf)} 段，其中可用 {len(usable)} 段\n")

# 实际文件
files = {p.stem: p for p in (D/"final").glob("*.npz")}
rows = []
for r in usable:
    p = files.get(r["name"])
    if not p:
        continue
    z = np.load(p)
    q = z["qpos"]
    T50 = int((len(q)-1) * (50.0/float(z["fps"]))) + 1   # 重采样后的帧数
    rows.append((r["name"], len(q), T50, r["seconds"], r["grade"]))

lens = np.array([r[2] for r in rows])
print(f"=== 重采样到 50Hz 后的长度分布（{len(rows)} 段）===")
for p in (0, 5, 25, 50, 75, 95, 100):
    print(f"  p{p:<4} {int(np.percentile(lens, p)):>7} 帧 "
          f"({np.percentile(lens, p)*0.02:>7.1f} 秒)")
print(f"  总计 {lens.sum():,} 帧 = {lens.sum()*0.02/60:.1f} 分钟")
print()
print(f"  最短: {min(rows, key=lambda r: r[2])[0]} -> {lens.min()} 帧")
print(f"  最长: {max(rows, key=lambda r: r[2])[0]} -> {lens.max()} 帧")
print()
print(f"!! 当前 load_reference 裁到 min={lens.min()} 帧：")
print(f"   保留 {lens.min()*len(rows):,} / {lens.sum():,} 帧 "
      f"= {lens.min()*len(rows)/lens.sum()*100:.1f}%")
print(f"   **浪费 {100-lens.min()*len(rows)/lens.sum()*100:.1f}% 的数据**")

print()
print("=== 动作类型分布（按名字前缀）===")
kinds = Counter(''.join(c for c in r[0].split('_')[0] if not c.isdigit())
                for r in rows)
for k, n in kinds.most_common():
    print(f"  {k:<22} {n:>3} 段")

print()
print("=== 最长的 12 段 ===")
for nm, t30, t50, sec, g in sorted(rows, key=lambda r: -r[2])[:12]:
    print(f"  {nm:<30} {t50:>6} 帧 {t50*0.02:>7.1f}s  {g}")

print()
print("=== 显存估算 ===")
NB = 10   # 1 anchor + 9 tracked
for n in (8, 16, 32, 68):
    sel = sorted(lens)[-n:] if n <= len(lens) else lens
    T = min(sel)                      # 当前实现的裁法
    padded = max(lens[:n]) if n <= len(lens) else lens.max()
    cur = n * T * (36 + 35 + NB*3) * 4 / 1e6
    full = n * padded * (36 + 35 + NB*3) * 4 / 1e6
    print(f"  {n:>3} 段: 裁到 min -> {cur:>7.1f} MB（{T} 帧）  "
          f"补齐到 max -> {full:>7.1f} MB（{padded} 帧）")
PYEOF
