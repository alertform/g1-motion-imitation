#!/usr/bin/env bash
python3 - <<'PYEOF'
import json, pathlib
p = pathlib.Path.home()/"tools"/"rl"/"runs"/"history.json"
h = json.loads(p.read_text())
print("冒烟测试曲线:")
for r in h["history"]:
    print(f"  {r['step']:>8} 步   奖励 {r['reward']:7.3f}   回合长 {r['ep_len']:6.1f}")
PYEOF
