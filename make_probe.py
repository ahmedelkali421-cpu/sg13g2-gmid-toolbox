#!/usr/bin/env python3
"""
=============================================================================
ADT-clone | ACCURACY PROBE, part 1 of 2
=============================================================================
Picks random points that fall BETWEEN grid nodes on all four axes, and
writes an ngspice control block that evaluates the real model at exactly
those points.

    python3 make_probe.py lut_nmos_hv_tt_27

writes
    probe_points.json    the point list, for part 2
    probe.spice          a COMPLETE netlist -- ngspice -b probe.spice

(this used to say probe_block.txt, a paste-into-xschem fragment. The
code has written a standalone netlist for a long time and says so on
its last line; the docstring was the only thing still promising a file
that is never created.)

Why off-grid on all four axes: a point sitting on a node would be read
straight out of the table and would prove nothing. The error we care
about is the interpolation error in the middle of a cell, which is where
it is largest.

The points are grouped by L so the script only pays for 20 resets rather
than one per point.
"""

import json
import os
import sys
import numpy as np

from adtlut import corner_of_table

N_L = 20          # distinct off-grid L values
N_PER_L = 12      # off-grid bias points at each L
MARGIN = 0.20     # keep this far inside the SOA so all 16 corners exist

# Not constants, for the same reason the CORNER below is not one: they
# are filled in main() from the table's own meta.json. Probing an hv
# table against the lv models would produce a believable page of
# numbers that means nothing, exactly as probing ss against tt would.
from make_sweep import (flavour_of, TEMP_NOM,          # noqa: E402
                        applied, body)

# Filled from the table's meta.json; an older table has no key.
TEMP = TEMP_NOM
# The flavour dict, filled in main(); applied()/body() read it.
_PF = None

CORE = DEVICE = None

# The saved and compared columns. This list went MISSING from this file
# at some point: COLS was referenced in four places -- the json, the
# .save block, the wrdata vector list and the per-case let block -- and
# defined in none, so the script died with NameError before writing
# anything. It is restored from the version that last ran, and it must
# stay a superset of every S["..."] that check_probe.py reads, because
# the two files agree only by this list.
#   leff is in here deliberately: it is the geometry read back out of
#   the model, which is how a wrong L would be caught rather than
#   assumed.
COLS = ["vgs", "vds", "vsb", "leff", "ids", "gm", "gds",
        "cgg", "cgsol", "cgdol", "fug", "vth", "vdss"]
PDK_LIB = None
# The corner is NOT a command-line argument and NOT a constant. It is
# derived from the table's own provenance: pack_lut.py records the raw
# file it packed in meta.json["source"], e.g. "lut_ss.raw". Probing an
# ss table against the tt models would produce a believable page of
# numbers that means nothing, and an argument is exactly the kind of
# thing that gets forgotten. If the corner cannot be derived, this
# refuses to write a netlist rather than guess one.


def mid_random(grid, rng, n, logspace=False, lo=None, hi=None):
    """n random values strictly inside random cells of `grid`."""
    g = np.asarray(grid, float)
    if lo is not None:
        g = g[g >= lo]
    if hi is not None:
        g = g[g <= hi]
    cell = rng.integers(0, len(g) - 1, n)
    f = rng.uniform(0.25, 0.75, n)          # stay away from the nodes
    if logspace:
        return np.exp(np.log(g[cell]) + f * (np.log(g[cell + 1])
                                             - np.log(g[cell])))
    return g[cell] + f * (g[cell + 1] - g[cell])


def main():
    global CORE, PDK_LIB, DEVICE
    lutdir = sys.argv[1] if len(sys.argv) > 1 else "lut_nmos_hv_tt_27"
    with open(os.path.join(lutdir, "meta.json")) as fh:
        m = json.load(fh)
    _, F = flavour_of(m["device"])
    global TEMP, _PF
    _PF = F
    TEMP = float(m.get("temp", TEMP_NOM))
    CORE, PDK_LIB, DEVICE = F["core"], F["lib"], F["device"]
    try:
        corner, src = corner_of_table(m, lutdir)
    except ValueError as e:
        sys.exit(str(e))
    print("=" * 70)
    print(f"  table   : {lutdir}")
    print(f"  built from : {src}")
    print(f"  so the probe will use the section : {corner}")
    print("=" * 70)
    Lg = np.asarray(m["L"], float)
    Gg = np.asarray(m["VGS"], float)
    Dg = np.asarray(m["VDS"], float)
    Bg = np.asarray(m["VSB"], float)
    vmax = float(m["vmax"])

    rng = np.random.default_rng(12345)

    # ------------------------------------------------------------------
    # Five groups. In each, one axis is SNAPPED ONTO grid nodes and the
    # other three stay between nodes. An axis that is snapped contributes
    # no interpolation error, so whichever group's error collapses tells
    # us which axis actually carries it. This replaces inferring the
    # culprit from a handful of worst-case rows.
    # ------------------------------------------------------------------
    GROUPS = [("all-off-grid", None),
              ("VGS-on-grid", "VGS"),
              ("VDS-on-grid", "VDS"),
              ("VSB-on-grid", "VSB"),
              ("L-on-grid", "L")]

    def pick(grid, lock, rng, lo=None, hi=None, logspace=False):
        g = np.asarray(grid, float)
        if lo is not None:
            g = g[g >= lo]
        if hi is not None:
            g = g[g <= hi]
        if len(g) < 2:
            return None
        if lock:                       # snap to an existing node
            return float(g[rng.integers(0, len(g))])
        return float(mid_random(g, rng, 1, logspace=logspace)[0])

    pts = []
    for gname, locked in GROUPS:
        per = N_PER_L if gname == "all-off-grid" else N_PER_L // 2
        if locked == "L":
            pool = Lg[2:-2] if len(Lg) > 6 else Lg
            n_L = min(N_L, len(pool))
            Ls = [float(x) for x in rng.choice(pool, n_L, replace=False)]
        else:
            Ls = list(mid_random(Lg, rng, N_L, logspace=True))
        for L in sorted(Ls):
            got, guard = 0, 0
            while got < per and guard < 3000:
                guard += 1
                vsb = pick(Bg, locked == "VSB", rng, hi=2.2)
                vgs = pick(Gg, locked == "VGS", rng, lo=0.1,
                           hi=vmax - vsb - MARGIN)
                vds = pick(Dg, locked == "VDS", rng, lo=0.05,
                           hi=vmax - vsb - MARGIN)
                if vgs is None or vds is None:
                    continue
                if vgs + vsb > vmax - MARGIN or vds + vsb > vmax - MARGIN:
                    continue
                pts.append(dict(L=float(L), VGS=vgs, VDS=vds, VSB=vsb,
                                group=gname))
                got += 1

    pts.sort(key=lambda p: p["L"])        # group by L to minimise resets
    Ls = np.array(sorted({p["L"] for p in pts}))

    with open("probe_points.json", "w") as fh:
        json.dump(dict(lutdir=lutdir, corner=corner, source=src,
                       cols=COLS, points=pts), fh, indent=1)

    # ---- the ngspice block -------------------------------------------
    # Emitted COMPLETE: .param, .save and .control together, so pasting it
    # as the whole content of the code symbol cannot drop the parameter
    # definitions the IHP subckt needs.
    out = []
    A = out.append
    A(f"* ADT-clone accuracy probe : {corner}")
    A("* standalone: ngspice -b probe.spice")
    A("")
    A(f'.lib {PDK_LIB} {corner}')
    A("")
    A(f".param w  = {m['W_nom']:.6g}")
    A(f".param l  = {m['L'][0]:.6g}")
    A(f".param ng = {m['ng']}")
    A("")
    # From the TABLE, never a flag: a probe at another temperature
    # would report the temperature difference as a table error.
    A(f".options TEMP={TEMP:g}")
    A("")
    A("* identical bias topology to the sweep that built the LUT:")
    A("*   S is the reference, B is driven negative")
    A("*   VGS = v(g)   VDS = v(d)   VSB = -v(b)")
    A("Vgs g 0 dc 0")
    A("Vds d 0 dc 0")
    A("Vs  s 0 dc 0")
    A("Vb  b 0 dc 0")
    A(f"XM1 d g s b {DEVICE} w={{w}} l={{l}} ng={{ng}} m=1")
    A("")
    for c in COLS:
        A(f".save {CORE}[{c}]")
    A(f".save {CORE}[weff]")
    A("")
    A(".control")
    A("")
    A("shell rm -f probe_spice.txt")
    A("")
    A("* ---- confirm this device is the SAME one the LUT was built on --")
    A("op")
    A(f"let c_nf = {CORE}[nf]")
    A(f"let c_we = {CORE}[weff]")
    A("let c_wt = c_nf * c_we")
    A("echo")
    A("echo ====== DEVICE CHECK ======")
    # The expected values come from THIS table. They used to be the
    # literals 5, 1.94e-06 and 9.7e-06 -- the HV NMOS reference device
    # -- echoed for every table, so on the other three device types the
    # check told the reader that a correct device was the wrong one.
    _ng = int(m["ng"])
    _wt = float(m["W_ref"])
    A(f"echo   nf      = $&c_nf      -- must be {_ng}")
    A(f"echo   weff    = $&c_we      -- must be {_wt / _ng:.4g}")
    A(f"echo   W total = $&c_wt      -- must be {_wt:.4g}")
    A("echo   if these differ, the probe device is not the LUT device")
    A("echo")
    A("")
    A("* Every number below is a literal. No shell variables are used,")
    A("* because this ngspice build does not substitute them inside")
    A("* wrdata or shell. Points are grouped by L so that the expensive")
    A("* alterparam + reset happens once per L, not once per point.")
    A("")
    vecs = " ".join("p_" + c for c in COLS)
    cur_L = None
    for p in pts:
        if p["L"] != cur_L:
            cur_L = p["L"]
            A("")
            A(f"alterparam l = {cur_L:.6e}")
            A("reset")
        # applied()/body(), never a bare sign. This file was one of
        # the five that carried its own copy of the n-type convention.
        # On a p table it drove the gate ABOVE the source, so all 240
        # points sat on a transistor that was off: ID = 0, which the
        # checker reported as a median error of exactly 100.000 %.
        A(f"alter Vgs dc = {applied(_PF, p['VGS']):.6f}")
        A(f"alter Vds dc = {applied(_PF, p['VDS']):.6f}")
        A(f"alter Vb  dc = {body(_PF, p['VSB']):.6f}")
        A("op")
        for c in COLS:
            A(f"let p_{c} = {CORE}[{c}]")
        A("set wr_singlescale")
        A("set appendwrite")
        A(f"wrdata probe_spice.txt {vecs}")
    A("")
    A("echo")
    A(f"echo ====== PROBE DONE : {len(pts)} points ======")
    A("echo   wrote probe_spice.txt")
    A("echo")
    A("")
    A("quit")
    A(".endc")
    A("")
    A(".end")

    with open("probe.spice", "w") as fh:
        fh.write("\n".join(out) + "\n")

    print(f"{len(pts)} probe points at {len(Ls)} off-grid L values")
    print(f"  L   range : {Ls[0]*1e6:.3f}u .. {Ls[-1]*1e6:.3f}u")
    print(f"  VGS range : {min(p['VGS'] for p in pts):.3f} .. "
          f"{max(p['VGS'] for p in pts):.3f} V")
    print(f"  VDS range : {min(p['VDS'] for p in pts):.3f} .. "
          f"{max(p['VDS'] for p in pts):.3f} V")
    print(f"  VSB range : {min(p['VSB'] for p in pts):.3f} .. "
          f"{max(p['VSB'] for p in pts):.3f} V")
    print()
    print("wrote probe_points.json  and  probe.spice")
    print()
    print("  run it with:   ngspice -b probe.spice")
    print("  no xschem needed, probe.spice is a complete netlist")


if __name__ == "__main__":
    main()
