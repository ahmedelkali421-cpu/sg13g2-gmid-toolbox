#!/usr/bin/env python3
"""
=============================================================================
ADT-clone | SWEEP GENERATOR  (Block B, as a standalone netlist)
=============================================================================
    python3 make_edge.py mos_tt

Writes edge_<corner>.spice : a COMPLETE netlist, no xschem needed.

WHY THIS SWEEP EXISTS
---------------------
The table scales every parameter with the number of fingers. Measured
against the model, that is exact for the channel quantities -- ID/ng is
constant to 0.0000 % over ng = 1 to 50 -- and WRONG for the overlap and
junction capacitances, because adjacent fingers SHARE their source and
drain diffusions: ng fingers have ng+1 junctions, not 2*ng. Those
parameters are affine in ng, not proportional:

    C(ng) = c*ng + e

Fitted to the measured Cgg_tot that form is accurate to 0.010 %, while
pure proportionality is out by 5.3 %. The error it leaves on the real
table is 1.4 % on Cgg_tot and fT, and 6.6 % on Cdtot -- the load
capacitance that sets a stage's output pole, so 6.6 % of its bandwidth.

Two finger counts determine both terms:

    c = (C5 - C1) / 4        e = (5*C1 - C5) / 4

so this is the same sweep at ng = 1. It is run on a COARSE voltage grid
on purpose: e is a 7 % correction, so 1 % accuracy on e is 0.07 % on the
result, and a 100 mV grid has 16x fewer points than the 25 mV one --
about half an hour instead of eight, and 60 MiB instead of 0.93 GiB.

    ngspice -b sweep_mos_ss.spice

Same sweep that built the TT table, same save list, same bias topology.
The only thing that changes between corners is the section on the .lib
line and the output file name, and both are written as literals, because
this ngspice build does not substitute shell variables inside write or
shell.
"""
import sys

# The device, the grid and the voltage ceiling all come from the same
# FLAVOUR table make_sweep.py builds the MAIN sweep from. They have to
# agree: attach_edge() refuses a pair whose L and VSB axes differ, and
# the affine split c = (P5 - P1)/4 is meaningless if the two tables were
# not taken at the same bias points. Importing is how that is enforced
# rather than hoped for.
from make_sweep import (FLAVOUR, PARAMS, CHECK, TAG, polarity,  # noqa: E402,F401
                        canon_corner,
                        _grid, TEMP_NOM, temp_tag, parse_temp)

VSB_N = 33

def build(corner, flav, temp=TEMP_NOM):
    F = FLAVOUR[flav]
    CORE, DEVICE = F["core"], F["device"]
    L_LIST = [f"{v:g}u" for v in _grid(F["l_spec"])]
    VSB_LIST = [f"{round(F['vsb_step']*i, 6):g}" for i in range(VSB_N)]
    VSTEP = F["edge_vstep"]
    # one DRAWN finger of the main sweep, so that the two tables differ
    # in their finger COUNT and in nothing else
    wf = float(F["w"].rstrip("u")) / int(F["ng"])
    raw = f"edge_{flav}_{TAG.get(corner, corner)}{temp_tag(temp)}.raw"
    o = []
    A = o.append
    A(f"* ADT-clone EDGE sweep, ng = 1, coarse grid : {DEVICE}  {corner}")
    A(f"* standalone : ngspice -b edge_{flav}_{corner}.spice")
    A(f"* output     : {raw}")
    A(f"* grid       : {len(L_LIST)} lengths x {len(VSB_LIST)} body biases,")
    A(f"*              VGS and VDS 0 to {F['vmax']} V in {VSTEP} V")
    A("")
    A(f".lib {F['lib']} {corner}")
    A("")
    A(f".param w  = {wf:g}u")
    A(".param l  = 1u")
    A(".param ng = 1")
    A("")
    # From the argument, never a literal: see make_sweep.py.
    A(f".options TEMP={temp:g}")
    A("")
    # polarity() from make_sweep, not a local copy. This file had no idea
    # p devices existed: it swept a pmos with the gate ABOVE the source,
    # so the transistor was off at all 862,092 points, and PSP -- which
    # normalises by negating -- reported vgs = -1.65 where the packer
    # wanted +1.65. The packer caught it on plot 0.
    pol = polarity(F)
    A(f"* S is the reference, B is driven "
      f"{'negative' if pol > 0 else 'POSITIVE'}, so")
    A(f"*   VGS = {'' if pol > 0 else '-'}v(g)   "
      f"VDS = {'' if pol > 0 else '-'}v(d)   "
      f"VSB = {'-' if pol > 0 else ''}v(b)")
    A("* and VGS, VDS can both go into one two-source dc sweep.")
    A("Vgs g 0 dc 0")
    A("Vds d 0 dc 0")
    A("Vs  s 0 dc 0")
    A("Vb  b 0 dc 0")
    A(f"XM1 d g s b {DEVICE} w={{w}} l={{l}} ng={{ng}} m=1")
    A("")
    for c in PARAMS + CHECK:
        A(f".save {CORE}[{c}]")
    A("")
    A("* NOTE on the control block below, kept comment-free on purpose:")
    A("*  - L is the OUTER loop. Only L needs alterparam + reset, and")
    A(f"*    reset is the expensive part: {len(L_LIST)} resets instead of")
    A(f"*    {len(L_LIST)*len(VSB_LIST)}.")
    A("*  - the dc line sweeps VGS and VDS together, one whole plane per")
    A("*    call, and vhi shrinks it with VSB so that VGB and VDB never")
    A(f"*    pass {F['vmax']} V. That is the incomplete grid: about 65 %")
    A("*    fewer points, dropped before being simulated, not after.")
    A("*  - filetype is set again before every write because reset can")
    A("*    clear it, and one silent ascii write would be a 20 GB file.")
    A("")
    A(".control")
    A("")
    A(f"shell rm -f {raw}")
    A("alter Vs dc = 0")
    A("")
    A(f"foreach Lv {' '.join(L_LIST)}")
    A("  alterparam l = $Lv")
    A("  reset")
    A(f"  foreach Sv {' '.join(VSB_LIST)}")
    # No leading '+' on the p branch: ngspice's preprocessor rejects
    # '+0.05' and silently SKIPS the alter, so the body never moves while
    # everything around it carries on. Sv is already positive.
    A(f"    alter Vb dc = {'-' if pol > 0 else ''}$Sv")
    A("    op")
    A(f"    let vhi = {F['vmax']} - $Sv")
    if pol > 0:
        A(f"    dc Vgs 0 $&vhi {VSTEP}  Vds 0 $&vhi {VSTEP}")
    else:
        A("    let vlo = -$&vhi")
        A(f"    dc Vgs 0 $&vlo -{VSTEP}  Vds 0 $&vlo -{VSTEP}")
    A("    set filetype = binary")
    A("    set appendwrite")
    A(f"    write {raw}")
    A("    destroy all")
    A("  end")
    A("  echo ---------------- one L finished ----------------")
    A("end")
    A("")
    A("echo")
    A(f"echo ====== EDGE SWEEP DONE : {corner} ======")
    A(f"echo   wrote {raw}")
    A("echo")
    A("")
    A("quit")
    A(".endc")
    A("")
    A(".end")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    args, temp = parse_temp(sys.argv[1:])
    if len(args) < 2 or args[0] not in FLAVOUR:
        sys.exit("usage: python3 make_edge.py <flavour> <corner> ... "
                 "[--temp C]\n"
                 "   e.g. python3 make_edge.py lv_p mos_tt --temp 125\n"
                 f"   flavours: {', '.join(FLAVOUR)}")
    flav = args[0]
    tt = temp_tag(temp)
    # same guard as make_sweep.py: an unknown corner is refused, not
    # written into the .lib line as typed
    corners = [canon_corner(c) for c in args[1:]]
    unknown = [c for c, cc in zip(args[1:], corners) if cc is None]
    if unknown:
        sys.exit(f"unknown corner {', '.join(unknown)}\n"
                 f"  give a section ({', '.join(TAG)}) or its short tag")
    for corner in corners:
        txt = build(corner, flav, temp)
        assert '"' not in txt, "double quote would break an xschem paste"
        fn = f"edge_{flav}_{corner}{tt}.spice"
        with open(fn, "w") as fh:
            fh.write(txt)
        print(f"wrote {fn}   ->  "
              f"edge_{flav}_{TAG.get(corner, corner)}{tt}.raw"
              f"   at {temp:g} C")
    print()
    print(f"run it with:   ngspice -b edge_{flav}_<corner>{tt}.spice")
    print()
    print("then pack it with the SAME packer, told the flavour and ng:")
    t = TAG.get(args[1], args[1])
    print(f"  python3 pack_lut.py edge_{flav}_{t}{tt}.raw "
          f"--flavour {flav} "
          f"--vstep {FLAVOUR[flav]['edge_vstep']} --ng 1")
