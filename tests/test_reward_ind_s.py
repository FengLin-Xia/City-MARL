import numpy as np
from logic.v5_reward_calculator import V5RewardCalculator
from contracts import ActionCandidate, EnvironmentState, RewardTerms


def fake_state(budget_ind=15000):
    class _State:
        budgets = {"IND": budget_ind}
    return _State()


def test_ind_s_example():
    import json, pathlib, os
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "city_config_v5_0.json"
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    calc = V5RewardCalculator(cfg)
    # 关闭 river/proximity 组件以避免外部依赖
    calc.components["river"] = False
    calc.components["proximity"] = False

    meta = {
        "zone": "near",
        "land_price_norm": 0.8,
        "river_dist_m": 30.0,
        "adj": 1
    }
    action = ActionCandidate(id=3, features=np.zeros(1), meta=meta)

    terms: RewardTerms = calc.calculate_reward(action, fake_state())

    # 成本应该为 1900
    assert abs(-terms.cost - 1900) < 1e-3
    # 月收益（revenue）在关闭 river / proximity 后应为 162
    assert abs(terms.revenue - 162) < 1e-3
