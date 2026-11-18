#!/usr/bin/env python3
"""
=============================================================================
ADT-clone | WHAT THE BUILT DEVICE DOES AT THE OTHER CORNERS
=============================================================================
    python3 adtcorner.py lut_nmos_lv_tt_27 VDS=0.825 VSB=0 GM/ID=15 L=0.5u ID=50u

Sizes at the NOMINAL table exactly as adtsize.py does, then freezes the
layout -- same L, same finger count, same drawn width -- and reads the
slow and fast tables at that one device.

WHY NOT JUST RUN adtsize THREE TIMES

Running the sizer once per corner answers 'what width would I have
needed if the wafer had come out slow'. It is a real number and it is
the wrong question, because a layout is drawn once, before anyone knows
which corner the wafer will be. Three answers of 16.44, 16.01 and 15.28
um cannot all be drawn.

Freezing the geometry and asking what it DOES is the question that gets
asked at the drawing stage, and the spread it reports is larger, because
at a fixed gm/ID the normalised variable absorbs most of the corner --
that is the whole argument for gm/ID design -- while a fixed layout does
not.

THE BIAS NETWORK IS NOT IN THE TABLE, SO BOTH BRACKETS ARE REPORTED

Freezing the geometry still leaves the question underdetermined: what is
held fixed at the gate?

  fixed VGS   the gate sits at a voltage the corner does not move: a
              resistive divider, or a voltage reference. The threshold
              shifts under it, so the CURRENT moves, and it moves a lot
              -- exponentially so in weak inversion.
  fixed ID    a current mirror forces the current and VGS follows. ID is
              then corner-independent by construction and the spread
              lands on gm, gm/ID and fT instead.

Real bias networks sit between the two, and which one a given stage is
closer to is a property of the schematic, not of the device. So this
prints both and does not pretend to know which applies. Read the column
that matches how the gate is actually driven.

WHAT IS AND IS NOT NEW HERE

No new simulation and no new physics: every number is read out of tables
that already exist, at a geometry the sizer already solved. The only new
thing is the question.
"""

import os
import sys

import numpy as np

from adtlut import LUT                                    # noqa: F401
from adtsize import NoSolution, Sizer, fmt, parse_value

# The corner tag lives in the directory name, by the same convention
# edge_dir_for uses for the companion table: lut_nmos_lv_tt_27 pairs
# with lut_nmos_lv_ss_27. Deriving it is what keeps the three tables of
# one device together without a second argument to forget.
CORNERS = ("ss", "tt", "ff")

# What a corner actually moves. ID, GM and the frequencies are the
# answer; VTH and VGS are there because they explain it.
SHOW = ["VTH", "VGS", "ID", "GM", "GM/ID", "GDS", "GM/GDS", "VDSAT",
        "FT", "VSTAR"]


def siblings(lutdir):
    """
    The same device's other corners, found by name.

    Returns [(corner, path)] for every one that exists on disk, in the
    order ss, tt, ff -- slow, nominal, fast -- so the printed columns
    read as a range and not as whatever order the filesystem gave.
    """
    base = os.path.basename(lutdir.rstrip("/"))
    parent = os.path.dirname(lutdir.rstrip("/"))
    # Two naming conventions are in use and both are legitimate:
    #   lut_nmos_hv_tt_27   the corner sits BETWEEN underscores
    #   lut_lv_p_tt         the corner ENDS the name
    # Matching only '_tt_' found the first and refused the second, which
    # is every pmos table. The substitution has to carry the same shape
    # back, or lut_lv_p_tt would look for lut_lv_p_ss_ and find nothing.
    tag = sep = None
    for c in CORNERS:
        if f"_{c}_" in base:
            tag, sep = c, "mid"
            break
        if base.endswith(f"_{c}"):
            tag, sep = c, "end"
            break
    if tag is None:
        sys.exit(f"  cannot tell the corner from the directory name "
                 f"{base!r}.\n"
                 f"  The corner has to appear as _tt_ / _ss_ / _ff_ inside "
                 f"the name, or as\n"
                 f"  _tt / _ss / _ff at the end of it, so that the three "
                 f"corners of one\n"
                 f"  device can be found without a second argument. "
                 f"Known: {', '.join(CORNERS)}.")
    out = []
    for c in CORNERS:
        if sep == "mid":
            name = base.replace(f"_{tag}_", f"_{c}_")
        else:
            name = base[: -len(tag)] + c
        p = os.path.join(parent, name)
        if os.path.isdir(p):
            out.append((c, p))
    return tag, out


def vgs_for_id(lut, target, L, VDS, VSB, W, ng, lo=None, hi=None):
    """
    The gate voltage that puts `target` amps through this exact device.

    ID is strictly monotone in VGS, so a bisection is safe and needs no
    starting guess beyond the table's own VGS axis. This is the
    current-mirror case: the geometry is fixed by the layout and the
    bias network sets the current, so VGS is the unknown.
    """
    def I(v):
        r = lut.eval("ID", L=np.array([L]), VGS=np.array([v]),
                     VDS=VDS, VSB=VSB, W=W, ng=ng)
        return float(np.asarray(r).ravel()[0])

    g = np.asarray(lut.VGS, float)
    lo = float(g[0]) if lo is None else lo
    hi = float(g[-1]) if hi is None else hi
    # walk in from the ends until both sides are finite: the incomplete
    # grid has no data above VMAX - VSB, and none below threshold worth
    # bisecting on
    step = (hi - lo) / 64.0
    while hi > lo and not np.isfinite(I(hi)):
        hi -= step
    while lo < hi and not np.isfinite(I(lo)):
        lo += step
    if not (np.isfinite(I(lo)) and np.isfinite(I(hi))):
        return float("nan")
    if not (I(lo) <= target <= I(hi)):
        return float("nan")          # the corner cannot reach it at all
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if I(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def read_at(sz, L, VGS, VDS, VSB, W, ng):
    """Every reported quantity at one device at one bias."""
    d = {}
    for k in SHOW:
        try:
            v = sz.lut.eval(k, L=np.array([L]), VGS=np.array([VGS]),
                            VDS=VDS, VSB=VSB, W=W, ng=ng)
            d[k] = float(np.asarray(v).ravel()[0])
        except Exception:                                # noqa: BLE001
            d[k] = float("nan")
    return d


UNIT = {"VTH": "V", "VGS": "V", "ID": "A", "GM": "S", "GDS": "S",
        "VDSAT": "V", "FT": "Hz", "VSTAR": "V", "GM/ID": "", "GM/GDS": ""}


def block(title, note, rows, nom_tag):
    """One table: a row per quantity, a column per corner, plus a spread."""
    o = [""]
    o.append("  " + "=" * 66)
    o.append(f"  {title}")
    o.append(f"  {note}")
    o.append("  " + "=" * 66)
    tags = [t for t, _ in rows]
    head = f"  {'quantity':<10}" + "".join(f"{t.upper():>14}" for t in tags)
    o.append(head + f"{'spread':>12}")
    o.append("  " + "-" * (len(head) + 10))
    for k in SHOW:
        vals = [d.get(k, float("nan")) for _, d in rows]
        cells = ""
        for v in vals:
            u = UNIT.get(k, "")
            cells += (f"{'--':>14}" if not np.isfinite(v)
                      else f"{fmt(v, u) if u else f'{v:.4g}':>14}")
        fin = [v for v in vals if np.isfinite(v) and v != 0]
        if len(fin) > 1 and min(fin) > 0:
            sp = f"{(max(fin)/min(fin) - 1)*100:+.1f} %"
        else:
            sp = "--"
        o.append(f"  {k:<10}{cells}{sp:>12}")
    o.append(f"  spread is max/min - 1 over the corners present; "
             f"{nom_tag.upper()} is the nominal.")
    return o


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    lutdir, args = argv[1], argv[2:]
    fixed, knobs = {"VSB": 0.0}, {}
    for a in args:
        if "=" not in a:
            sys.exit(f"expected name=value, got {a!r}")
        k, v = a.split("=", 1)
        if k.upper() in ("VDS", "VSB"):
            fixed[k.upper()] = parse_value(v)
        elif k.lower() == "ng":
            sys.exit("ng is an output of the sizer, not an input")
        else:
            knobs[k] = parse_value(v)
    if "VDS" not in fixed:
        sys.exit("VDS is required: it is part of the bias plan, not a spec")

    nom_tag, sibs = siblings(lutdir)
    VDS, VSB = fixed["VDS"], fixed["VSB"]

    # ---- size at the nominal table, exactly as adtsize does -----------
    sz = Sizer(lutdir)
    try:
        r = sz.solve(VDS=VDS, VSB=VSB, **knobs)
    except (ValueError, NoSolution) as e:
        print(f"  NO SOLUTION at {os.path.basename(lutdir)}: {e}")
        return 2

    L = float(r["L"])
    ng = int(r["ng"])
    W = float(r["W"])                 # DRAWN, the built device
    VGS0 = float(r["VGS"])
    ID0 = float(r["snapped"]["ID"])

    print("=" * 70)
    print(f"  CORNER SPREAD OF ONE BUILT DEVICE")
    print("=" * 70)
    print(f"  sized on  : {os.path.basename(lutdir)}")
    print(f"  asked for : " + "  ".join(f"{k}={v:g}" if isinstance(v, float)
                                        else f"{k}={v}"
                                        for k, v in knobs.items()))
    print(f"  built     : L = {fmt(L,'m')}   {ng} fingers   "
          f"W = {fmt(W,'m')} drawn")
    print(f"  bias      : VDS = {fmt(VDS,'V')}   VSB = {fmt(VSB,'V')}   "
          f"VGS = {fmt(VGS0,'V')} at {nom_tag.upper()}")
    print(f"  corners   : " + ", ".join(t.upper() for t, _ in sibs))
    if len(sibs) < 2:
        print()
        print("  only one corner table is on disk, so there is no spread")
        print("  to report. Build the others with make_sweep.py and")
        print("  pack them beside this one.")
        return 0

    # ---- the two brackets ---------------------------------------------
    rowsV, rowsI, notes = [], [], []
    for tag, path in sibs:
        s2 = Sizer(path)
        rowsV.append((tag, read_at(s2, L, VGS0, VDS, VSB, W, ng)))
        v = vgs_for_id(s2.lut, ID0, L, VDS, VSB, W, ng)
        if not np.isfinite(v):
            notes.append(f"  {tag.upper()}: cannot reach "
                         f"{fmt(ID0,'A')} at this geometry and VDS, so "
                         f"its fixed-current column is blank. A mirror "
                         f"would not hold at this corner either.")
            rowsI.append((tag, {}))
        else:
            rowsI.append((tag, read_at(s2, L, v, VDS, VSB, W, ng)))

    for ln in block(
            "A. GATE HELD AT A FIXED VOLTAGE",
            "a divider or a reference drives the gate; the corner moves "
            "VTH under it",
            rowsV, nom_tag):
        print(ln)
    for ln in block(
            "B. DRAIN CURRENT HELD FIXED",
            "a current mirror sets ID and VGS follows the threshold",
            rowsI, nom_tag):
        print(ln)
    for n in notes:
        print(n)

    print()
    print("  Neither column is 'the' answer. A real bias network sits")
    print("  between them, so these bracket the device and the schematic")
    print("  decides which end it is nearer. Read A for a gate driven")
    print("  from a voltage, B for a gate driven from a mirror.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
