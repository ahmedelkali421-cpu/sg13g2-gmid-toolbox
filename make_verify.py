#!/usr/bin/env python3
"""
=============================================================================
ADT-clone | END-TO-END VERIFICATION, part 1 of 2
=============================================================================
    python3 make_verify.py lut_nmos_hv_tt_27
    ngspice -b verify.spice
    python3 check_verify.py

Everything checked so far has been checked against itself. The report is
consistent with the table, and the table is consistent with the model. What
nobody has asked is the only question a designer cares about:

    this device -- L = 565.9 nm, W = 66.9 um, ng = 5, at VGS = 702.6 mV --
    does it actually give 100 uA, 1.5 mS and 2 GHz?

That is a question about the MODEL, not about the table. This closes the
chain: specification -> bias -> geometry -> silicon.

It runs the sizing assistant on a spread of specifications, takes the
(L, W, ng, VGS) it produces, and writes a standalone netlist that
simulates exactly those devices at exactly that bias. Part 2 compares
three things:

  1. SPICE against the SPECIFICATION   -- did the device meet the ask
  2. SPICE against the SIZER           -- did the table mispredict anything
  3. snapped against ideal, in SPICE   -- what the quantisation really cost

It also re-checks nf and weff on every case. The total-W-versus-W-per-finger
confusion was the first silent bug in this project, and ng is now chosen
automatically and lands anywhere from 5 to 25, so the door is open again.
"""

import json
import os
import sys

import numpy as np

from adtlut import corner_of_table

from adtsize import Sizer, NoSolution

# Not constants. main() fills them from the table's own meta.json, by
# the same reasoning that already makes the CORNER come from the table
# rather than from an argument: a netlist written to check a table must
# simulate the device that table was built from, and a device name
# typed here is a thing that gets forgotten.
from make_sweep import (flavour_of, polarity,           # noqa: E402
                        applied, body, TEMP_NOM)

CORE = PDK_LIB = DEVICE = None

# EVERY stored parameter is saved, not just the handful the report shows.
# Two of these cases exist only to measure how each parameter scales with
# W, and that needs all of them. leff and weff are the geometry read back
# from the model itself, which is how a wrong W is caught rather than
# assumed.
PARAMS = ["ids", "gm", "gmb", "gds", "vth", "vgt", "vdss", "cgg", "cgs",
          "cgd", "cgb", "cdg", "cdb", "csg", "csb", "cgsol", "cgdol",
          "cjd", "cjs", "fug", "vearly", "beff", "rg", "sfl", "sid",
          "sig", "fknee", "cigid", "igs", "igd", "igb", "iavl", "ijd"]
CHECK = ["vgs", "vds", "vsb", "weff", "leff"]
COLS = PARAMS + CHECK

# ---------------------------------------------------------------------------
# THREE geometry studies, all at one L and one bias, because the end-to-end
# run showed ID off by up to 40 % while Cgg, VTH and gm/ID were right to
# about 1 %. Current moves, capacitance and threshold do not: that is a
# MOBILITY effect, and the only geometry it can be tracking is the width
# per FINGER. The table is characterised at 1.94 um of effective width per
# finger and the sized devices sat at 4.3 to 49.6 um.
#
#   A  constant finger width, ng = 1..50
#      Every device is the reference finger repeated. This is the
#      assumption the table actually rests on, and it is the one that was
#      never tested. ID/ng must be constant.
#
#   B  constant total EFFECTIVE width, ng = 1..25
#      Same channel area, different finger width. Note EFFECTIVE, not
#      drawn: weff_total = W_drawn - ng*dW, so holding the DRAWN width
#      while varying ng shrinks the real channel by up to 3 % and that
#      shows up in ID and in Cgg together -- a dW artefact that reads
#      exactly like a layout effect. The drawn width is set per ng so
#      that weff_total is identical, and the current DENSITY is what
#      gets compared.
#
#   C  constant ng, width scaled
#      The study already run. It conflates the two and is kept only to
#      show that it does.
# ---------------------------------------------------------------------------
STUDY = dict(L=1.0e-6, VGS=1.1, VDS=0.9, VSB=0.0)
# The table's reference finger, filled in main() from its meta.json.
# It was the literal 2.0e-6, which is the hv table's. Run against the
# lv table, whose finger is 1 um, study A simulated a device the table
# has no prediction for, so section 4D of check_verify.py -- the one
# test of the finger-count laws against spice -- printed its header,
# its separator lines and not a single row, without saying why.
WF_DRAWN = None
# Filled in main() from the table's meta.json. A table packed before TEMP
# was recorded has no key, and that means the nominal.
TEMP = TEMP_NOM
NG_A = [1, 2, 5, 10, 25, 50]
# 50 was added so that study B reaches a 1 um finger. At the fixed 50 um
# total below, ng = 25 gives a 2 um finger, which IS the hv table's own
# reference -- so for hv the study measured the finger-width effect right
# down to the width the table holds. The lv table's finger is 1 um, one
# step narrower than the study went, so its figure was a FLOOR and not a
# measurement: the effect grows as the finger narrows. check_verify.py
# now says which of the two it has, and with ng = 50 here it has the
# measurement on both tables.
NG_B = [1, 2, 5, 10, 25, 50]
W_TOTAL_B = 50.0e-6          # total EFFECTIVE width, held fixed
MULT_C = [1.0, 2.0, 4.0]
NG_C = 5
W_BASE_C = 20.0e-6

# A spread, not a demo: both ends of the inversion range, the shortest and
# a long L, zero and non-zero body bias, and the four cases already run by
# hand so their printed numbers can be checked against silicon.
SPECS = [
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "FT": 2e9, "ID": 100e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 10, "L": 1e-6, "ID": 100e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 20, "L": 2e-6, "ID": 20e-6}),
    dict(VDS=0.9, VSB=0.5, knobs={"ID": 50e-6, "GM": 750e-6, "L": 1.5e-6}),
    dict(VDS=1.2, VSB=0.0, knobs={"GM/ID": 5, "L": 0.45e-6, "ID": 500e-6}),
    dict(VDS=0.6, VSB=0.0, knobs={"GM/ID": 25, "L": 1e-6, "ID": 10e-6}),
    dict(VDS=1.5, VSB=1.0, knobs={"GM/ID": 12, "L": 0.8e-6, "ID": 200e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 18, "L": 5e-6, "ID": 5e-6}),
    # ---- added after the first run with the finger feedback on -------
    # That run left exactly one specification above 0.03 % after the
    # rounding was accounted for -- and it was spec 0, which was BOTH
    # the only one carrying an fT target AND the only one whose L landed
    # inside a cell instead of on a node. Two candidate causes, one
    # sample: not separable. Section 2 said it was the L axis, because
    # fT's miss (-0.340 %) equalled gm's (-0.342 %) while Cgg_tot sat at
    # -0.002 %, but that is an inference and these three make it a
    # measurement. Six extra operating points, seconds of ngspice.
    #
    #   8  fT with L PINNED to a stored node: fT's own accuracy, with
    #      the L-axis interpolation taken out of the question entirely.
    #   9  a second fT target, far from the first in inversion level
    #      and in current, so one point is not the whole evidence.
    #  10  Cdtot as the width knob, L on a node. Cdtot is the one
    #      quantity still above 0.1 % (median 0.152 %, max 0.658 %) and
    #      it is built from the affine terms, so this says whether that
    #      residual is the coarse 100 mV edge grid or the L axis.
    dict(VDS=0.9, VSB=0.0, knobs={"FT": 2e9, "L": 1e-6, "ID": 100e-6}),
    # 5 GHz at gm/ID = 12 is simply not reachable on this HV device --
    # the solver refused it and printed the range it CAN reach along the
    # first constraint, which is the feasible-region answer and the right
    # one. Replaced with a second fT target that is reachable, at an L
    # that is also a stored node, so it measures fT and not the L axis.
    dict(VDS=0.9, VSB=0.0, knobs={"FT": 3e9, "L": 0.6e-6, "ID": 200e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "L": 1e-6,
                                  "CDTOT": 100e-15}),
    # ---- added after the double-root fix -----------------------------
    # The one case the tool answered WRONGLY for its whole life, and
    # never said so.
    #
    # gm/ID is not monotone in VGS: it is 1/(n.Ut), and n = 1 + Cdep/Cox
    # falls as the depletion region widens, so gm/ID rises through weak
    # inversion, peaks at the weak/moderate boundary and only then
    # falls. Measured here at VDS = 0.9, VSB = 0: the 0.45 um device
    # starts at 24.72 at VGS = 0 and peaks at 28.12. A target of 25 is
    # therefore crossed TWICE -- at VGS = 0.0209 climbing and at VGS =
    # 0.5757 falling -- and _vgs_curve took the first crossing it found
    # scanning upward, which is the climbing one.
    #
    # There the device carries 4e-8 A/m against the other branch's
    # 0.21 A/m. For ID = 10 uA that is a device 252 METRES wide. The
    # report printed it without comment, because nothing in the chain
    # knew a second solution existed.
    #
    # Every other specification here is single-rooted: targets below
    # ~24 sit under the curve's own starting value at VGS = 0, so the
    # climbing branch never reaches them, and lengths of 0.8 um and up
    # start above 26.5. Only a high target at the shortest L has two.
    # So this spec, and no other, tests the fix. It should now come
    # back at VGS near 0.576 and about 24 fingers.
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 25, "L": 0.45e-6,
                                  "ID": 10e-6}),
    # ---- added after l_axis_probe flagged the L axis ------------------
    # Every L above is a STORED NODE, on purpose: spec 8's comment says
    # so outright -- 'L PINNED to a stored node ... with the L-axis
    # interpolation taken out of the question entirely'. That was the
    # right choice for measuring fT and it is a blind spot for measuring
    # the L axis, because at a stored node the interpolation error is
    # zero BY CONSTRUCTION. The only free-L case is spec 0.
    #
    # l_axis_probe measured, by leave-one-out and with no simulation,
    # that sg13_hv_nmos carries up to 1.078 % on ID, gm and gds in the
    # cells between 0.50 and 0.85 um, and that five of those eight cells
    # are a genuine KINK -- the reverse short-channel hump, which a
    # 50 nm grid straddles rather than resolves. Identically at tt, ss
    # and ff: the hump is the device's, not the corner's.
    #
    # That number is an upper bound the TABLE computed on ITSELF. These
    # four put it in front of ngspice, at the MIDPOINT of each flagged
    # cell -- the farthest point from a node, where the error is largest,
    # instead of on a node where it is zero:
    #
    #   0.525 um   between 0.50 and 0.55   bound 1.078 %
    #   0.575 um   between 0.55 and 0.60   bound 1.078 %
    #   0.675 um   between 0.65 and 0.70   bound 0.321 %
    #   0.775 um   between 0.75 and 0.80   bound 0.615 %
    #
    # and spec 9 at 0.6 um stays the CONTROL: a stored node, which must
    # come back at the 0.00x % the rest of the suite reports. If the
    # midpoints land near 1 % and the control does not, the bound is
    # honest and a 25 nm refinement over 0.45 to 0.90 is worth its sweep
    # time. If the midpoints land at 0.01 % too, the bound is
    # conservative by two orders and the refinement buys nothing.
    #
    # On the lv table 0.525 and 0.575 are themselves stored nodes, since
    # that grid steps by 25 nm from 0.30 to 0.60, so there those two
    # degenerate into two more node controls. That is not a waste: lv
    # measured 0 kinks and <= 0.31 %, so lv is where the suite should
    # keep agreeing, and the flagged cells are hv's alone.
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "L": 0.525e-6,
                                  "ID": 50e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "L": 0.575e-6,
                                  "ID": 50e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "L": 0.675e-6,
                                  "ID": 50e-6}),
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "L": 0.775e-6,
                                  "ID": 50e-6}),
    # The four above are hv's flagged cells. lv's worst cell is somewhere
    # else entirely -- the probe flags 0.22 to 0.25 um, where that grid
    # steps by 10 nm and not 50 -- and NONE of the cases above goes near
    # it: the two lv cases that did land inside a cell sat at 0.675 and
    # 0.775 um, which lv resolves at 50 nm without trouble. So the lv
    # column of the node-versus-cell split was measuring a length its own
    # bound was not about, and the two numbers were not comparable.
    #
    # 0.235 um is the midpoint of the 0.23 to 0.24 cell. It is below hv's
    # shortest gate, so on an hv table the sizer refuses it and the spec
    # is SKIPPED with its reason printed -- the same way spec 6 is skipped
    # on lv for exceeding the SOA. A case that only one flavour can run is
    # not a problem; a flavour with no case in its own worst cell is.
    dict(VDS=0.9, VSB=0.0, knobs={"GM/ID": 15, "L": 0.235e-6,
                                  "ID": 50e-6}),
]

# what the sizer predicts and SPICE will be asked about
# CDTOT and CGD are in here because they are built from the AFFINE
# parameters (cjd, cgdol) and so carry the finger end term. Cdtot is the
# load capacitance that sets a stage's bandwidth, so if the end term
# matters anywhere it matters there, and it had not been checked.
PREDICT = ["ID", "GM", "GDS", "GM/ID", "GM/GDS", "VTH", "VDSAT",
           "CGG_TOT", "FT", "VSTAR", "VA", "CDTOT", "CGD"]


def main():
    global CORE, PDK_LIB, DEVICE, WF_DRAWN, TEMP
    lutdir = sys.argv[1] if len(sys.argv) > 1 else "lut_nmos_hv_tt_27"
    with open(os.path.join(lutdir, "meta.json")) as fh:
        m = json.load(fh)
    _, F = flavour_of(m["device"])
    CORE, PDK_LIB, DEVICE = F["core"], F["lib"], F["device"]
    WF_DRAWN = m["W_nom"] / m["ng"]
    TEMP = float(m.get("temp", TEMP_NOM))
    try:
        corner, src = corner_of_table(m, lutdir)
    except ValueError as e:
        sys.exit(str(e))
    print("=" * 72)
    print(f"  table      : {lutdir}  (packed from {src})")
    print(f"  models     : {corner}   at {TEMP:g} C "
          f"(from the table, not a flag)")
    print("=" * 72)

    s = Sizer(lutdir)
    if s.edge_dir:
        print(f"  edge table : {s.edge_dir}  (affine finger correction ON)")
    else:
        print("  edge table : none found, so the overlap and junction")
        print("               capacitances are scaled as proportional")
    # fT targets are made RELATIVE to this device. A fixed 2 GHz is a
    # mild ask of a fast LV NMOS and impossible for a slow HV PMOS, so a
    # fixed number either refused the spec or sized a metres-wide device
    # on the weaker flavours (spec 0 on HV PMOS wanted 0.4 fingers and
    # rounded to 1, a +150 % artifact the report then blamed on fT). The
    # ceiling is the fastest this device runs at the spec's own bias;
    # the target is a fraction of it, so every flavour gets a reachable
    # fT at the same inversion level.
    import numpy as _np

    def ft_at(L, gmid, VDS, VSB):
        """fT at a REALISTIC operating point: this L and this gm/ID.

        The peak fT is at the top of the voltage range, a density so
        high that at a fixed ID it needs a sliver of a finger -- which
        is exactly the absurd sizing the fixed 2 GHz produced. Anchoring
        instead to a moderate gm/ID gives an fT every flavour reaches at
        a width a real design would use.
        """
        vhi = float(s.lut.meta["vmax"]) - VSB
        vg = _np.linspace(0.1, vhi - 0.05, 160)
        gi = _np.asarray(s.lut.eval("GM/ID", L=L, VGS=vg, VDS=VDS,
                                    VSB=VSB), float)
        ft = _np.asarray(s.lut.eval("FT", L=L, VGS=vg, VDS=VDS, VSB=VSB),
                         float)
        ok = _np.isfinite(gi) & _np.isfinite(ft)
        if not ok.any():
            return _np.nan
        k = _np.nanargmin(_np.abs(gi[ok] - gmid))
        return float(ft[ok][k])

    SPECS_D = []
    for sp in SPECS:
        kn = dict(sp["knobs"])
        if "FT" in kn:
            # L to anchor at: the spec's own pinned L, else a mid length
            # so the fT-selected L lands inside the range, not at an end
            Lref = kn.get("L", float(s.lut.L[min(3, len(s.lut.L) - 1)]))
            # gm/ID to anchor at: moderate inversion (10), unless the
            # spec pins its own gm/ID -- then honour it
            gmid = kn.get("GM/ID", 10.0)
            ft = ft_at(Lref, gmid, sp["VDS"], sp["VSB"])
            if _np.isfinite(ft) and ft > 0:
                kn["FT"] = round(ft, -6) or ft
        SPECS_D.append(dict(sp, knobs=kn))

    cases = []
    for i, sp in enumerate(SPECS_D):
        try:
            r = s.solve(VDS=sp["VDS"], VSB=sp["VSB"], **sp["knobs"])
        except (NoSolution, ValueError) as e:
            print(f"  spec {i}: SKIPPED, {str(e).splitlines()[0][:54]}")
            continue
        # Only the BUILT device is simulated. The 'asked for' column of
        # the report is a fictitious continuous width whose finger width
        # is not the reference one, and simulating that would reintroduce
        # the very error this change removes.
        cases.append(dict(spec=i, knobs=sp["knobs"], VDS=sp["VDS"],
                          VSB=sp["VSB"], L=r["L"], VGS=r["VGS"],
                          kind="built", W=r["W"], ng=r["ng"],
                          ng_real=r["ng_real"], ng_err=r["ng_err"],
                          pred={k: r["snapped"].get(k) for k in PREDICT},
                          asked={k: r["ideal"].get(k) for k in PREDICT}))
        knb = "  ".join(f"{k}={v:g}" for k, v in sp["knobs"].items())
        print(f"  spec {i}: {knb}")
        print(f"          -> L={r['L']*1e9:.1f}n  VGS={r['VGS']*1e3:.1f}m"
              f"  {r['ng']} x {r['wf_drawn']*1e6:g}u = "
              f"{r['W']*1e6:.3f}u   (wanted {r['ng_real']:.3f} fingers, "
              f"{r['ng_err']:+.2f} %)")

    if not cases:
        sys.exit("no specification produced a device; nothing to verify")

    st = STUDY
    def add(kind, W, ng):
        cases.append(dict(spec=-1, knobs={}, VDS=st["VDS"], VSB=st["VSB"],
                          L=st["L"], VGS=st["VGS"], kind=kind,
                          W=W, ng=ng, pred={}))
    for ng in NG_A:
        add(f"A/ng={ng}", ng * WF_DRAWN, ng)
    for ng in NG_B:
        # drawn width chosen so the total EFFECTIVE width is the same
        add(f"B/ng={ng}", W_TOTAL_B + ng * s.lut.dW, ng)
    for mu in MULT_C:
        add(f"C/x{mu:g}", mu * W_BASE_C, NG_C)
    print(f"  study A: finger {WF_DRAWN*1e6:g}u drawn, ng = "
          f"{', '.join(map(str, NG_A))}")
    print(f"  study B: total Weff {W_TOTAL_B*1e6:g}u FIXED, ng = "
          f"{', '.join(map(str, NG_B))}")
    print(f"  study C: ng = {NG_C}, W = "
          f"{', '.join(f'{m*W_BASE_C*1e6:g}u' for m in MULT_C)}")
    print(f"  all at L={st['L']*1e6:g}u VGS={st['VGS']:g} "
          f"VDS={st['VDS']:g} VSB={st['VSB']:g}")

    with open("verify_cases.json", "w") as fh:
        # the stored L axis goes with the cases, so part 2 can say
        # whether each solution landed on a node or inside a cell
        json.dump(dict(lutdir=lutdir, corner=corner, source=src,
                       cols=COLS, predict=PREDICT, L_axis=m["L"],
                       cases=cases), fh, indent=1)

    o = []
    A = o.append
    A(f"* ADT-clone end-to-end verification : {corner}")
    A("* standalone : ngspice -b verify.spice")
    A("")
    A(f".lib {PDK_LIB} {corner}")
    A("")
    A(f".param w  = {cases[0]['W']:.9e}")
    A(f".param l  = {cases[0]['L']:.9e}")
    A(f".param ng = {cases[0]['ng']}")
    A("")
    # From the TABLE's meta.json, never a flag. A verify run at a
    # different temperature than the table was built at measures the
    # temperature difference and reports it as a table error.
    A(f".options TEMP={TEMP:g}")
    A("")
    # The table holds MAGNITUDES, because PSP normalises what it reports.
    # The netlist has to apply the signed terminal voltages, so every
    # value below goes through applied() and body() from make_sweep --
    # the same two functions the sweep itself used. make_edge.py learned
    # this the hard way: it had its own copy of the n-type convention and
    # swept a pmos with the gate ABOVE the source, 862,092 points of a
    # transistor that was off.
    _F = flavour_of(DEVICE)[1]
    A("* same bias topology as the sweep that built the table:")
    A(f"*   S is the reference, B is driven "
      f"{'negative' if polarity(_F) > 0 else 'POSITIVE'}")
    A("Vgs g 0 dc 0")
    A("Vds d 0 dc 0")
    A("Vs  s 0 dc 0")
    A("Vb  b 0 dc 0")
    A(f"XM1 d g s b {DEVICE} w={{w}} l={{l}} ng={{ng}} m=1")
    A("")
    for c in COLS:
        A(f".save {CORE}[{c}]")
    A("")
    A("* w, l and ng are all netlist parameters, so each case needs")
    A("* alterparam plus reset; nf is read with let, which works for a")
    A("* device parameter that .save does not accept.")
    A("")
    A(".control")
    A("")
    A("shell rm -f verify_spice.txt")
    vecs = " ".join("p_" + c for c in COLS) + " p_nf"
    for k, c in enumerate(cases):
        A("")
        A(f"alterparam w  = {c['W']:.9e}")
        A(f"alterparam l  = {c['L']:.9e}")
        A(f"alterparam ng = {c['ng']}")
        A("reset")
        A(f"alter Vgs dc = {applied(_F, c['VGS']):.9f}")
        A(f"alter Vds dc = {applied(_F, c['VDS']):.9f}")
        A(f"alter Vb  dc = {body(_F, c['VSB']):.9f}")
        A("op")
        for cc in COLS:
            A(f"let p_{cc} = {CORE}[{cc}]")
        A(f"let p_nf = {CORE}[nf]")
        A("set wr_singlescale")
        A("set appendwrite")
        A(f"wrdata verify_spice.txt {vecs}")
    A("")
    A("echo")
    A(f"echo ====== VERIFY DONE : {len(cases)} devices ======")
    A("echo   wrote verify_spice.txt")
    A("echo")
    A("")
    A("quit")
    A(".endc")
    A("")
    A(".end")
    txt = "\n".join(o) + "\n"
    assert '"' not in txt
    with open("verify.spice", "w") as fh:
        fh.write(txt)

    print()
    nb = sum(1 for c in cases if c["kind"] == "built")
    print(f"  {len(cases)} devices to simulate "
          f"({nb} from specifications, {len(cases)-nb} geometry studies)")
    print("  wrote verify_cases.json  and  verify.spice")
    print()
    print("  next:  ngspice -b verify.spice")
    print("  then:  python3 check_verify.py")


if __name__ == "__main__":
    main()
