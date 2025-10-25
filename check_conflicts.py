import re
from collections import defaultdict

# 读取所有月份文件
slots = defaultdict(list)  # key: (x, y), value: [(month, agent_id, line_content)]

for month in range(1, 31):
    filename = f'outputs/export_month_{month:02d}.txt'
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        continue
    
    # 解析每一行
    for line in content.split('\n'):
        if not line.strip():
            continue
        
        # 匹配所有动作
        matches = re.finditer(r'(\d+)\(([0-9.]+),([0-9.]+)', line)
        for match in matches:
            agent_id = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            slot_key = (x, y)
            slots[slot_key].append((month, agent_id, line))

# 找出冲突的槽位
conflicts = {k: v for k, v in slots.items() if len(v) > 1}

print(f"Total {len(conflicts)} slot conflicts")
print(f"Total {len(slots)} unique slots\n")

# 显示前10个冲突
for i, (slot, occurrences) in enumerate(list(conflicts.items())[:10]):
    print(f"Conflict {i+1}: Slot ({slot[0]:.1f},{slot[1]:.1f})")
    for month, agent_id, line in occurrences:
        print(f"  Month {month}, Agent {agent_id}: {line[:50]}...")
