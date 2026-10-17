#!/usr/bin/env python3
"""Sample the B60's hwmon power/temperature/fan counters while a workload runs.

Read-only and root-free (hwmon8 exposes energy1/energy2 in uJ, tempN_input in mC),
so this can run unprivileged while the cap sweep script owns the sudo writes.

Usage:
    phase9_sampler.py --hwmon /sys/class/drm/card0/device/hwmon/hwmon8 \
                      --out phase9_150w.csv --duration 40 [--interval 0.25]
"""
from __future__ import annotations

import argparse
import csv
import os
import time


def read_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hwmon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--interval", type=float, default=0.25)
    args = ap.parse_args()

    h = args.hwmon
    fields = ["t_s", "card_W", "pkg_W", "temp_pkg_mC", "temp_mctrl_mC",
              "temp_pcie_mC", "temp_vram_mC", "fan_rpm"]
    cap = read_int(os.path.join(h, "power1_cap"))

    rows: list[dict[str, float]] = []
    t0 = time.time()
    e1 = read_int(os.path.join(h, "energy1_input"))
    e2 = read_int(os.path.join(h, "energy2_input"))
    t_prev = t0

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        while time.time() - t0 < args.duration:
            now = time.time()
            n1 = read_int(os.path.join(h, "energy1_input"))
            n2 = read_int(os.path.join(h, "energy2_input"))
            if n1 is not None and e1 is not None and now > t_prev:
                dt = now - t_prev
                row = {
                    "t_s": round(now - t0, 2),
                    "card_W": (n1 - e1) / 1e6 / dt,
                    "pkg_W": ((n2 - e2) / 1e6 / dt) if (n2 is not None and e2 is not None) else "",
                    "temp_pkg_mC": read_int(os.path.join(h, "temp2_input")) or "",
                    "temp_mctrl_mC": read_int(os.path.join(h, "temp4_input")) or "",
                    "temp_pcie_mC": read_int(os.path.join(h, "temp5_input")) or "",
                    "temp_vram_mC": read_int(os.path.join(h, "temp3_input")) or "",
                    "fan_rpm": read_int(os.path.join(h, "fan1_input")) or "",
                }
                w.writerow(row)
                fh.flush()
                rows.append(row)
                e1, e2, t_prev = n1, n2, now
            time.sleep(args.interval)

    card = [float(r["card_W"]) for r in rows if r["card_W"] != ""]
    pkg = [float(r["pkg_W"]) for r in rows if r["pkg_W"] != ""]
    print(f"[sampler] {len(rows)} samples over {args.duration:.0f}s "
          f"| cap = {cap // 1000000 if cap else '?'} W")
    if card:
        mid = sorted(card)[len(card) // 2]
        print(f"[sampler] card W  min/median/max = {min(card):.1f} / {mid:.1f} / {max(card):.1f}"
              + (f"   ({100 * max(card) / (cap / 1e6):.0f}% of cap)" if cap else ""))
    if pkg:
        print(f"[sampler] pkg  W  min/median/max = {min(pkg):.1f} / {sorted(pkg)[len(pkg) // 2]:.1f} / {max(pkg):.1f}")
    for key, name in (("temp_pkg_mC", "pkg"), ("temp_mctrl_mC", "mctrl"),
                      ("temp_pcie_mC", "pcie"), ("temp_vram_mC", "vram")):
        vals = [int(r[key]) for r in rows if r[key] != ""]
        if vals:
            print(f"[sampler] temp {name:6s} max = {max(vals) / 1000:.1f} C")
    fans = [int(r["fan_rpm"]) for r in rows if r["fan_rpm"] != ""]
    if fans:
        print(f"[sampler] fan max = {max(fans)} RPM")
    print(f"[sampler] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
