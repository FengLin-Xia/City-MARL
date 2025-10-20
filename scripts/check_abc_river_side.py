import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_slots_xy(slots_path: str) -> Dict[str, Tuple[float, float]]:
    xy: Dict[str, Tuple[float, float]] = {}
    with open(slots_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except Exception:
                continue
            xy[f"s_{idx}"] = (x, y)
    return xy


def main():
    cfg = load_cfg('configs/city_config_v4_1.json')
    rl_agents = cfg.get('solver', {}).get('rl', {}).get('agents', ['IND', 'EDU'])
    # Slots and river center line (simple side classifier)
    slots_path = cfg.get('growth_v4_1', {}).get('slots', {}).get('path', 'slots_with_angle.txt')
    slot_xy = load_slots_xy(slots_path)
    rivers: List[Dict[str, Any]] = cfg.get('terrain_features', {}).get('rivers', [])
    coords = rivers[0].get('coordinates', []) if rivers else []
    all_y = [pt[1] for pt in coords if isinstance(pt, list) and len(pt) >= 2]
    center_y = sum(all_y) / len(all_y) if all_y else 0.0

    # EDU expected side by hub Y (same side = y>center or y<center)
    hubs = cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
    try:
        edu_idx = rl_agents.index('EDU')
    except ValueError:
        edu_idx = 0
    edu_hub_y = hubs[edu_idx][1] if edu_idx < len(hubs) else hubs[0][1]
    edu_side_is_above = (edu_hub_y > center_y)

    with open('models/v4_1_rl/slot_selection_history.json', 'r', encoding='utf-8') as f:
        hist = json.load(f)

    counts = {
        'A': {'same': 0, 'other': 0},
        'B': {'same': 0, 'other': 0},
        'C': {'same': 0, 'other': 0},
    }

    total_by_size = defaultdict(int)

    for ep in hist.get('episodes', []):
        for step in ep.get('steps', []):
            if step.get('agent') != 'EDU':
                continue
            for da in step.get('detailed_actions', []):
                size = da.get('size')
                if size not in ('A', 'B', 'C'):
                    continue
                fps = da.get('footprint_slots', []) or []
                if not fps:
                    continue
                sid = fps[0]
                xy = slot_xy.get(sid)
                if xy is None:
                    continue
                slot_above = (xy[1] > center_y)
                if slot_above == edu_side_is_above:
                    counts[size]['same'] += 1
                else:
                    counts[size]['other'] += 1
                total_by_size[size] += 1

    print('EDU A/B/C placements by river side (same vs other side):')
    for size in ('A', 'B', 'C'):
        same = counts[size]['same']
        other = counts[size]['other']
        tot = same + other
        pct_other = (other / tot * 100.0) if tot > 0 else 0.0
        print(f"  {size}: same={same}, other={other}, total={tot}, other_side%={pct_other:.2f}%")


if __name__ == '__main__':
    main()


