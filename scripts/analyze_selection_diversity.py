import json
import math
import os
from collections import Counter, defaultdict


def flatten_selected_slots(selected_slots):
    flat = []
    for group in selected_slots:
        # Each group is like ["s_123"]
        for slot_id in group:
            flat.append(slot_id)
    return flat


def compute_metrics(counter: Counter):
    total = sum(counter.values())
    unique = len(counter)
    if total == 0 or unique == 0:
        return {
            "total": total,
            "unique": unique,
            "entropy_bits": 0.0,
            "effective_num_choices": 0.0,
            "hhi": 0.0,
            "top": []
        }

    probs = [count / total for count in counter.values()]
    entropy_bits = -sum(p * math.log(p, 2) for p in probs if p > 0)
    effective_num_choices = 2 ** entropy_bits  # perplexity
    hhi = sum(p * p for p in probs)

    top = counter.most_common(10)
    top_with_pct = [
        {
            "slot_id": slot_id,
            "count": count,
            "pct": round(100.0 * count / total, 3)
        }
        for slot_id, count in top
    ]

    return {
        "total": total,
        "unique": unique,
        "entropy_bits": entropy_bits,
        "effective_num_choices": effective_num_choices,
        "hhi": hhi,
        "top": top_with_pct,
    }


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    json_path = os.path.join(repo_root, "models", "v4_1_rl", "slot_selection_history.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall_counter: Counter = Counter()
    agent_counters: defaultdict[str, Counter] = defaultdict(Counter)
    size_counter: Counter = Counter()
    agent_size_counters: defaultdict[str, Counter] = defaultdict(Counter)

    episodes = data.get("episodes", [])
    for ep in episodes:
        for step in ep.get("steps", []):
            agent = step.get("agent", "UNKNOWN")
            selected_slots = step.get("selected_slots", [])
            flat_slots = flatten_selected_slots(selected_slots)
            if not flat_slots:
                continue
            overall_counter.update(flat_slots)
            agent_counters[agent].update(flat_slots)

            # sizes from detailed actions
            for da in step.get("detailed_actions", []):
                sz = str(da.get("size", "")).upper()
                if sz:
                    size_counter.update([sz])
                    agent_size_counters[agent].update([sz])

    overall_metrics = compute_metrics(overall_counter)

    print("=== Overall Diversity ===")
    print(f"Total selections: {overall_metrics['total']}")
    print(f"Unique slots: {overall_metrics['unique']}")
    print(f"Entropy (bits): {overall_metrics['entropy_bits']:.4f}")
    print(f"Effective number of choices (2^H): {overall_metrics['effective_num_choices']:.2f}")
    print(f"HHI (lower is more diverse): {overall_metrics['hhi']:.6f}")
    print("Top 10 slots:")
    for item in overall_metrics["top"]:
        print(f"  {item['slot_id']}: {item['count']} ({item['pct']}%)")

    print("\n=== Diversity by Agent ===")
    for agent, counter in sorted(agent_counters.items()):
        m = compute_metrics(counter)
        print(f"\nAgent: {agent}")
        print(f"  Total selections: {m['total']}")
        print(f"  Unique slots: {m['unique']}")
        print(f"  Entropy (bits): {m['entropy_bits']:.4f}")
        print(f"  Effective number of choices (2^H): {m['effective_num_choices']:.2f}")
        print(f"  HHI: {m['hhi']:.6f}")
        print("  Top 5 slots:")
        for item in m["top"][:5]:
            print(f"    {item['slot_id']}: {item['count']} ({item['pct']}%)")

    # Size distributions
    total_actions = sum(size_counter.values())
    ordered_sizes = ["S", "M", "L", "A", "B", "C"]

    print("\n=== Size Distribution (overall) ===")
    print(f"Total actions: {total_actions}")
    for sz in ordered_sizes:
        cnt = size_counter.get(sz, 0)
        pct = (100.0 * cnt / total_actions) if total_actions > 0 else 0.0
        print(f"  {sz}: {cnt} ({pct:.2f}%)")

    print("\n=== Size Distribution by Agent ===")
    for agent in sorted(agent_size_counters.keys()):
        c = agent_size_counters[agent]
        subtot = sum(c.values())
        print(f"Agent: {agent}  (Total actions: {subtot})")
        for sz in ordered_sizes:
            cnt = c.get(sz, 0)
            pct = (100.0 * cnt / subtot) if subtot > 0 else 0.0
            print(f"  {sz}: {cnt} ({pct:.2f}%)")


if __name__ == "__main__":
    main()


