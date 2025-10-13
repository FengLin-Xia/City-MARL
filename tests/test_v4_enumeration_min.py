from logic.v4_enumeration import SlotNode, V4Planner, _auto_fill_neighbors_4n


def build_grid_slots(w: int, h: int):
    slots = {}
    for y in range(h):
        for x in range(w):
            sid = f"s_{x}_{y}"
            slots[sid] = SlotNode(slot_id=sid, x=x, y=y)
    _auto_fill_neighbors_4n(slots)
    return slots


def main():
    slots = build_grid_slots(4, 4)
    candidates = set(slots.keys())
    occupied = set()

    import math
    max_d = math.hypot(3, 3)

    def lp_provider(sid: str) -> float:
        n = slots[sid]
        d = math.hypot(n.x, n.y)
        v = 1.0 - d / max_d
        return max(0.0, min(1.0, v))

    cfg = {
        "growth_v4_0": {
            "enumeration": {
                "length_max": 3,
                "beam_width": 8,
                "max_expansions": 200,
                "caps": {
                    "top_slots_per_agent_size": {"EDU": {"S": 50}, "IND": {"L": 10}}
                },
                "objective": {
                    "EDU": {"w_r": 0.3, "w_p": 0.6, "w_c": 0.1},
                    "IND": {"w_r": 0.6, "w_p": 0.2, "w_c": 0.2},
                    "normalize": "per-month-pool-minmax",
                },
            }
        }
    }

    planner = V4Planner(cfg)
    actions, best_seq = planner.plan(
        slots=slots,
        candidates=candidates,
        occupied=occupied,
        lp_provider=lp_provider,
        agent_types=["EDU", "IND"],
        sizes={"EDU": ["S", "M", "L"], "IND": ["S", "M", "L"]},
    )

    print("Total actions:", len(actions))
    print("Sample actions (first 8):")
    for a in actions[:8]:
        print(a.agent, a.size, a.footprint_slots, f"lp={a.LP_norm:.2f}", f"score={a.score:.3f}")

    print("\nBest sequence:")
    print("Length:", len(best_seq.actions), "Score:", round(best_seq.score, 3))
    for i, a in enumerate(best_seq.actions, 1):
        print(i, a.agent, a.size, a.footprint_slots, f"score={a.score:.3f}")


if __name__ == "__main__":
    main()



