"""
Minimal FinanceSystem for v3.6/v4.0 integration.

Provides:
- FinanceSystem
  - initialize_building_finance(state)
  - calculate_monthly_finance(month, land_price_field, heat_field)
  - save_building_finance_csv(output_dir, month)
  - save_finance_dashboard(output_dir, quarter)
  - save_kpi_summary_csv(output_dir, quarter)

This implementation is intentionally lightweight and self-contained,
so the simulation can run even without external finance configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import csv
import json


@dataclass
class BuildingFinance:
    id: str
    type: str
    x: float
    y: float
    construction_cost: float = 0.0
    operating_cost: float = 0.0
    last_revenue: float = 0.0
    last_profit: float = 0.0


class FinanceSystem:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.buildings: Dict[str, BuildingFinance] = {}
        self._monthly_rows: List[Dict] = []

    # ------------------- Public API -------------------
    def initialize_building_finance(self, state: Dict):
        """Initialize or refresh building finance records from simulation state.
        State format expected:
          state = { 'residential': [...], 'commercial': [...], 'industrial': [...], 'public': [...] }
          each building: {'id': str, 'type': str, 'xy': [x, y], ...}
        """
        for bt in ['residential', 'commercial', 'industrial', 'public']:
            for b in state.get(bt, []):
                bid = str(b.get('id'))
                if not bid:
                    continue
                xy = b.get('xy', [0.0, 0.0])
                x = float(xy[0]) if len(xy) > 0 else 0.0
                y = float(xy[1]) if len(xy) > 1 else 0.0
                # Reasonable defaults per type
                cons, opex = self._defaults_for_type(bt)
                self.buildings[bid] = BuildingFinance(
                    id=bid, type=bt, x=x, y=y,
                    construction_cost=cons, operating_cost=opex
                )

    def calculate_monthly_finance(self, month: int, land_price_field, heat_field=None) -> Dict:
        """Compute simple revenue/profit for all buildings.
        revenue is influenced by land price at (x,y) and base per type.
        """
        records = []
        for bf in self.buildings.values():
            # Fetch land price if available (2D numpy array with origin lower-left assumed by caller)
            lp_val = 0.0
            try:
                if land_price_field is not None:
                    h = land_price_field.shape[0]
                    y_idx = int(max(0, min(h - 1, round(bf.y))))
                    x_idx = int(max(0, min(land_price_field.shape[1] - 1, round(bf.x))))
                    lp_val = float(land_price_field[y_idx, x_idx])
            except Exception:
                lp_val = 0.0
            base_rev = self._base_revenue_for_type(bf.type)
            rev = base_rev * (0.7 + 0.6 * lp_val)
            opex = bf.operating_cost
            profit = rev - opex

            bf.last_revenue = rev
            bf.last_profit = profit
            row = {
                'month': int(month),
                'id': bf.id,
                'type': bf.type,
                'location_x': bf.x,
                'location_y': bf.y,
                'revenue': float(rev),
                'opex': float(opex),
                'profit': float(profit),
            }
            records.append(row)

        # Keep rows for saving CSV
        self._monthly_rows = records
        return { 'month': month, 'num_buildings': len(records) }

    def save_building_finance_csv(self, output_dir: str, month: int):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'building_finance_month_{int(month):02d}.csv')
        header = ['month', 'id', 'type', 'location_x', 'location_y', 'revenue', 'opex', 'profit']
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in self._monthly_rows:
                w.writerow(r)

    def save_finance_dashboard(self, output_dir: str, quarter: int):
        os.makedirs(output_dir, exist_ok=True)
        # Aggregate simple totals per type
        agg: Dict[str, Dict[str, float]] = {}
        for bf in self.buildings.values():
            a = agg.setdefault(bf.type, { 'revenue': 0.0, 'profit': 0.0 })
            a['revenue'] += bf.last_revenue
            a['profit'] += bf.last_profit
        out = { 'quarter': int(quarter), 'by_type': agg }
        path = os.path.join(output_dir, f'finance_dashboard_quarterly_{int(quarter):02d}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def save_kpi_summary_csv(self, output_dir: str, quarter: int):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'kpi_summary_quarterly_{int(quarter):02d}.csv')
        header = ['quarter', 'type', 'total_revenue', 'total_profit']
        # Same aggregation as dashboard
        agg: Dict[str, Dict[str, float]] = {}
        for bf in self.buildings.values():
            a = agg.setdefault(bf.type, { 'revenue': 0.0, 'profit': 0.0 })
            a['revenue'] += bf.last_revenue
            a['profit'] += bf.last_profit
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(header)
            for t, v in agg.items():
                w.writerow([int(quarter), t, float(v['revenue']), float(v['profit'])])

    # ------------------- Helpers -------------------
    def _defaults_for_type(self, t: str) -> (float, float):
        # construction_cost, operating_cost (very rough defaults)
        t = (t or '').lower()
        if t == 'residential':
            return 1.0, 0.20
        if t == 'commercial':
            return 1.6, 0.32
        if t == 'industrial':
            return 1.8, 0.28
        if t == 'public':
            return 2.0, 0.40
        return 1.2, 0.25

    def _base_revenue_for_type(self, t: str) -> float:
        t = (t or '').lower()
        if t == 'residential':
            return 0.50
        if t == 'commercial':
            return 0.85
        if t == 'industrial':
            return 0.95
        if t == 'public':
            return 0.30
        return 0.40



