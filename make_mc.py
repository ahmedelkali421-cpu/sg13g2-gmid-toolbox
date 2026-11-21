#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | MONTE CARLO AGAINST THE MISMATCH FORMULA, part 1 of 2
=============================================================================
    python3 make_mc.py lut_nmos_hv_tt_27            (1000 samples)
    python3 make_mc.py lut_nmos_hv_tt_27 3000
    ngspice -b mc.spice
    python3 check_mc.py

The mismatch block of the sizing report is arithmetic on four Pelgrom
coefficients. Two things in it are ASSUMED and not measured: that the four
sources add in quadrature, and that the length term is the table's
dlnID/dlnL times 3 nm. This asks the model.

THREE DEVICES, CHOSEN SO EACH TERM IS TESTED WHERE IT DOMINATES

    A   L = 0.45 um, gm/ID = 25    the length term is 68 % of the total,
                                   through dlnID/dlnL = -5.25
    B   L = 1.1 um,  gm/ID = 25    dlnID/dlnL = -0.24, so the length term
                                   all but vanishes and the threshold
                                   term is alone
    C   L = 1 um,    gm/ID = 10    a mixture, in strong inversion

A formula that is right at all three is right for three different
reasons. One that is right at only C would have passed any single test.

All three sit in ONE circuit, so one reset draws all of them and the run
is a third as long. They share nothing but the drain supply.

HOW A SAMPLE IS DRAWN, AND THE TRAP IN IT

The PDK puts agauss() on the device line inside the subcircuit, so a new
value is drawn each time the netlist is PARSED -- not each time it is
simulated. An op after an op is the same device. So every sample is

    setseed k   ->   reset   ->   op

and the reset is what makes it a new device. The seed is set explicitly
so that sample k is the same device tomorrow.

WHAT IS SAVED, AND WHY IT IS MORE THAN THE CURRENT

ids alone would give one number, sigma(ID), to hold against the formula.
vth, weff and leff are saved with it, because the model reports its own
perturbed values: sigma(leff) is S_DL measured directly, sigma(vth)
checks A_VT, and regressing ln(ID) on ln(leff) across the samples
MEASURES dlnID/dlnL -- the number the whole length term rests on --
without touching the table at all.

THE NOMINAL DEVICE IS RUN LAST

With mm_ok = 0 the PDK calls agauss(x, s, 0), a sigma count of zero.
Nothing in this project has exercised that branch, and if it misbehaves
the Monte Carlo data should already be on disk. It goes to its own file
and the checker treats it as optional.
"""

import json
import os
import sys

from gmid_lut import corner_of_table
from gmid_size import Sizer, NoSolution

# Not constants. main() fills them from the table's own meta.json, by
# the same reasoning that already makes the CORNER come from the table
# rather than from an argument.
from make_sweep import (flavour_of, TEMP_NOM,          # noqa: E402
                        applied, body)

# Filled from the table's meta.json; an older table has no key.
TEMP = TEMP_NOM
# The flavour dict, filled in main(); applied()/body() read it.
_PF = None

PDK_LIB = DEVICE = None
SAVE = ["ids", "gm", "vth", "weff", "leff"]

POINTS = [
    dict(name="A", why="length term dominates",
         VDS=0.9, VSB=0.0, knobs={"GM/ID": 25, "L": 0.45e-6, "ID": 10e-6}),
    dict(name="B", why="threshold term alone",
         VDS=0.9, VSB=0.0, knobs={"GM/ID": 25, "L": 1.1e-6, "ID": 10e-6}),
    dict(name="C", why="mixture, strong inversion",
         VDS=0.9, VSB=0.0, knobs={"GM/ID": 10, "L": 1.0e-6, "ID": 100e-6}),
]


def core(k):
    return f"@n.xm{k}.n{DEVICE}"


def main():
    global PDK_LIB, DEVICE
    lutdir = sys.argv[1] if len(sys.argv) > 1 else "lut_nmos_hv_tt_27"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    with open(os.path.join(lutdir, "meta.json")) as fh:
        m = json.load(fh)
    _, F = flavour_of(m["device"])
    global TEMP, _PF
    _PF = F
    TEMP = float(m.get("temp", TEMP_NOM))
    PDK_LIB, DEVICE = F["lib"], F["device"]
    try:
        corner, src = corner_of_table(m, lutdir)
    except ValueError as e:
        sys.exit(str(e))
    lib = corner + "_mismatch"

    s = Sizer(lutdir)
    if not s.mm:
        sys.exit("  " + s.mm.why_not)

    print("=" * 72)
    print(f"  table   : {lutdir}")
    print(f"  models  : {lib}   (the table's corner plus mismatch)")
    print(f"  samples : {N}")
    print("=" * 72)

    devs = []
    for p in POINTS:
        try:
            r = s.solve(VDS=p["VDS"], VSB=p["VSB"], **p["knobs"])
        except (NoSolution, ValueError) as e:
            sys.exit(f"  point {p['name']} has no solution: "
                     f"{str(e).splitlines()[0]}")
        sn = r["snapped"]
        devs.append(dict(
            name=p["name"], why=p["why"], VDS=p["VDS"], VSB=p["VSB"],
            L=r["L"], W=r["W"], ng=r["ng"], VGS=r["VGS"],
            pred={k: sn.get(k) for k in
                  ("ID", "GM", "GM/ID", "VTH", "SIGVT", "SIGBETA", "SIGW",
                   "SIGL", "SIGID", "_dlnI_dlnL", "_area")}))
        print(f"  {p['name']}: L={r['L']*1e6:g}u  W={r['W']*1e6:g}u  "
              f"ng={r['ng']}  VGS={r['VGS']*1e3:.1f}m   "
              f"sigma(ID) predicted {sn['SIGID']:.3f} %   ({p['why']})")

    with open("mc_cases.json", "w") as fh:
        json.dump(dict(lutdir=lutdir, corner=lib, N=N, save=SAVE,
                       S_DL=s.mm.d["S_DL"], S_DW=s.mm.d["S_DW"],
                       devices=devs), fh, indent=1)

    o = []
    A = o.append
    A(f"* SG13G2 gm/ID Toolbox mismatch Monte Carlo : {lib} : {N} samples")
    A("* standalone : ngspice -b mc.spice")
    A("")
    A(f".lib {PDK_LIB} {lib}")
    A("")
    A(".param mmok = 1")
    # From the TABLE, never a flag: a probe at another temperature
    # would report the temperature difference as a table error.
    A(f".options TEMP={TEMP:g}")
    A("")
    A("Vds d 0 dc 0")
    A("Vs  s 0 dc 0")
    A("Vb  b 0 dc 0")
    for k, d in enumerate(devs, 1):
        A(f"Vg{k} g{k} 0 dc 0")
        A(f"XM{k} d g{k} s b {DEVICE} w={d['W']:.9e} l={d['L']:.9e} "
          f"ng={d['ng']} m=1 mm_ok={{mmok}}")
    A("")
    for k in range(1, len(devs) + 1):
        for c in SAVE:
            A(f".save {core(k)}[{c}]")
    A("")
    A(".control")
    A("")
    A("shell rm -f mc_spice.txt mc_nominal.txt")
    A("set wr_singlescale")
    A("set appendwrite")
    vecs = " ".join(f"q{k}_{c}" for k in range(1, len(devs) + 1)
                    for c in SAVE)

    def block(out):
        # applied()/body(), never a bare sign: this file was the last of
        # the five carrying its own copy of the n-type convention.
        A(f"alter Vds dc = {applied(_PF, devs[0]['VDS']):.9f}")
        A(f"alter Vb  dc = {body(_PF, devs[0]['VSB']):.9f}")
        for k, d in enumerate(devs, 1):
            A(f"alter Vg{k} dc = {applied(_PF, d['VGS']):.9f}")
        A("op")
        for k in range(1, len(devs) + 1):
            for c in SAVE:
                A(f"let q{k}_{c} = {core(k)}[{c}]")
        A(f"wrdata {out} {vecs}")
        A("destroy all")

    for i in range(1, N + 1):
        A("")
        A(f"setseed {i}")
        A("reset")
        block("mc_spice.txt")
        if i % 100 == 0:
            A(f"echo   sample {i} of {N}")

    A("")
    A("* the nominal device, LAST: mm_ok = 0 is an untested branch")
    A("alterparam mmok = 0")
    A("reset")
    block("mc_nominal.txt")
    A("")
    A("echo")
    A(f"echo ====== MC DONE : {N} samples x {len(devs)} devices ======")
    A("echo   wrote mc_spice.txt and mc_nominal.txt")
    A("echo")
    A("quit")
    A(".endc")
    A("")
    A(".end")
    txt = "\n".join(o) + "\n"
    assert '"' not in txt
    with open("mc.spice", "w") as fh:
        fh.write(txt)

    print()
    print(f"  wrote mc_cases.json and mc.spice "
          f"({len(o)} lines, {N} resets)")
    print("  every sample re-reads the model library, so this is the")
    print("  slowest deck in the project per data point. It prints a")
    print("  line every 100 samples.")
    print()
    print("  next:  ngspice -b mc.spice")
    print("  then:  python3 check_mc.py")


if __name__ == "__main__":
    main()
