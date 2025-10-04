"""
Export slots with displacement angle from a deform run directory.

Reads: <input_dir>/displacements.csv
Writes: <input_dir>/slots_with_angle.txt

Angle definition:
- zero_axis='x': 0° points to +X, angle = atan2(dy, dx) in degrees, mapped to [0, 360)
- zero_axis='y': 0° points to +Y, angle = atan2(dx, dy) in degrees, mapped to [0, 360)
"""

from __future__ import annotations

import argparse
import csv
import math
import os


def export_slots_with_angle(
    input_dir: str,
    csv_name: str = "displacements.csv",
    output_name: str = "north_slots_with_angle.txt",
    zero_axis: str = "x",
    write_header: bool = False,
    include_id: bool = False,
) -> str:
    inp = os.path.join(input_dir, csv_name)
    outp = os.path.join(input_dir, output_name)
    if not os.path.exists(inp):
        raise FileNotFoundError(f"not found: {inp}")

    with open(inp, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(outp, "w", encoding="utf-8") as g:
        if write_header:
            if include_id:
                g.write("slot_id,x,y,angle_deg\n")
            else:
                g.write("x,y,angle_deg\n")
        for row in rows:
            try:
                idx = int(row.get("orig_index", 0))
                nx = float(row.get("new_x", 0.0))
                ny = float(row.get("new_y", 0.0))
                dx = float(row.get("dx", 0.0))
                dy = float(row.get("dy", 0.0))
            except Exception:
                # skip malformed
                continue
            if zero_axis.lower().startswith("y"):
                ang = math.degrees(math.atan2(dx, dy)) % 360.0
            else:
                ang = math.degrees(math.atan2(dy, dx)) % 360.0
            if include_id:
                g.write(f"s_{idx},{nx:.6f},{ny:.6f},{ang:.2f}\n")
            else:
                g.write(f"{nx:.6f},{ny:.6f},{ang:.2f}\n")
    return outp


def main():
    p = argparse.ArgumentParser(description="Export slots with angle from displacements.csv")
    p.add_argument("--input_dir", required=True, help="directory containing displacements.csv")
    p.add_argument("--csv", default="displacements.csv", help="csv file name inside input_dir")
    p.add_argument("--output", default="slots_with_angle.txt", help="output txt file name inside input_dir")
    p.add_argument("--zero_axis", choices=["x", "y"], default="x", help="0° direction: x or y")
    p.add_argument("--header", action="store_true", help="write header line")
    p.add_argument("--include_id", action="store_true", help="include slot_id in the first column")
    args = p.parse_args()

    outp = export_slots_with_angle(
        args.input_dir,
        args.csv,
        args.output,
        args.zero_axis,
        write_header=bool(args.header),
        include_id=bool(args.include_id),
    )
    print(f"[export] saved: {outp}")


if __name__ == "__main__":
    main()


