#!/usr/bin/env python3
"""
=============================================================================
ADT-clone | SWEEP GENERATOR  (Block B, as a standalone netlist)
=============================================================================
    python3 make_sweep.py mos_ss

Writes sweep_<corner>.spice : a COMPLETE netlist, no xschem needed.

    ngspice -b sweep_mos_ss.spice

Same sweep that built the TT table, same save list, same bias topology.
The only thing that changes between corners is the section on the .lib
line and the output file name, and both are written as literals, because
this ngspice build does not substitute shell variables inside write or
shell.
"""
import sys

PARAMS = ["ids", "gm", "gmb", "gds", "vth", "vgt", "vdss", "cgg", "cgs",
          "cgd", "cgb", "cdg", "cdb", "csg", "csb", "cgsol", "cgdol",
          "cjd", "cjs", "fug", "vearly", "beff", "rg", "sfl", "sid",
          "sig", "fknee", "cigid", "igs", "igd", "igb", "iavl", "ijd"]
CHECK = ["vgs", "vds", "vsb", "weff", "leff"]

MODELS = "/foss/pdks/ihp-sg13g2/libs.tech/ngspice/models"


def _grid(spec):
    """
    An L list written as (stop, step) pairs, in MICRONS, starting at the
    first stop. The short end carries the fine steps because that is
    where the threshold moves with length; the long end is flat and a
    coarse step there costs nothing.

    Returns floats. pack_lut.py imports this function and this table's
    l_spec rather than keeping a second copy of the list: the packer
    reconstructs the grid from the L axis, so a list that disagreed with
    the one the sweep ran would mis-shelve every plot, silently.
    """
    out, x = [], spec[0][0]
    out.append(x)
    for stop, step in spec[1:]:
        while x < stop - 1e-9:
            x = round(x + step, 6)
            out.append(x)
    return out


def grid_m(spec):
    """
    The same list in METRES, built by parsing each value's own shortest
    decimal rather than by multiplying by 1e-6.

    Accumulating round(x+step, 6) and then scaling lands up to one ULP
    away from the double a literal like 10.0e-6 gives, and 10.0*1e-6 is
    a different double from 10.0e-6 outright. That is 1e-22 m and could
    not matter physically, but meta.json is written once and read for
    the life of the table, and 'the L the netlist swept' and 'the L the
    table says it swept' should be the same number and not merely a very
    close one. Formatting to the shortest decimal and parsing it back
    gives exactly the double a hand-written literal would.
    """
    return [float(f"{v:g}e-6") for v in _grid(spec)]


# ---------------------------------------------------------------------------
# The two flavours. Every number here is from the SG13G2 Process
# Specification Rev 1.2, not from a guess:
#
#   HV  section 2.4   VGS <= 3.3 V,   L from 0.45 um (VTNHV10x045) to
#                     10 um (VTNHV10x10),  Wmin 0.30 um
#   LV  section 2.1   VGS <= 1.65 V,  L from 0.13 um (VTN10x013) to
#                     10 um (VTN10x10),    Wmin 0.15 um
#
# VMAX sets the SOA clip: the dc sweep stops at VMAX - VSB so that
# neither VGB nor VDB ever passes the rating. That is the incomplete
# grid, and it is what makes the table storable.
#
# VSTEP and the VSB step are chosen so that BOTH tables hold 133 voltage
# points and 33 body-bias points. The LV device runs at half the voltage,
# so it gets half the step, and the two tables then have the same
# RELATIVE resolution rather than the same absolute one.
#
# L_SPEC is denser for LV on purpose. The spec gives VTN10x013 = 0.50 V
# and VTN10x10 = 0.20 V: the threshold moves 300 mV with length. The HV
# device moves 10 mV over its own range (0.70 to 0.69). An L axis that
# resolves the HV device is nowhere near enough for the LV one.
# ---------------------------------------------------------------------------
FLAVOUR = {
    "hv": dict(
        device="sg13_hv_nmos",
        core="@n.xm1.nsg13_hv_nmos",
        lib=f"{MODELS}/cornerMOShv.lib",
        vmax="3.3", vstep="0.025", edge_vstep="0.1",
        vsb_step=0.1,
        w="10u", ng="5",                       # 2 um fingers
        l_spec=[(0.45, 0), (1.0, 0.05), (2.0, 0.1), (3.0, 0.2),
                (6.0, 0.5), (10.0, 1.0)],
    ),
    "lv": dict(
        device="sg13_lv_nmos",
        core="@n.xm1.nsg13_lv_nmos",
        lib=f"{MODELS}/cornerMOSlv.lib",
        vmax="1.65", vstep="0.0125", edge_vstep="0.05",
        vsb_step=0.05,
        w="5u", ng="5",                        # 1 um fingers
        l_spec=[(0.13, 0), (0.30, 0.01), (0.60, 0.025), (1.0, 0.05),
                (2.0, 0.1), (3.0, 0.2), (6.0, 0.5), (10.0, 1.0)],
    ),
    # ---- p type ------------------------------------------------------
    # 'pol' is the ONLY field that distinguishes a p device anywhere in
    # this project, and it is read in exactly one place: the netlist this
    # file writes. It multiplies the applied terminal voltages, and it
    # flips the body source, because reverse body bias puts the well
    # ABOVE the source for a p device and BELOW it for an n device.
    #
    # Nothing downstream needs it. pmos_probe.py measured why: the PSP
    # model card carries type = -1 and the model reports every quantity
    # already normalised -- applied VGS = -0.5 comes back as vgs = +0.5,
    # ids, vth, gm, gmb, gds all positive. So the rawfile that reaches
    # pack_lut.py is the same shape as an nmos one, and adtlut, adtsize,
    # adtcorner, make_mismatch and check_verify need no p-type branch at
    # all. Had the model reported signed values, the flip map would have
    # had to live in adtlut AND agree with check_verify AND fake_model --
    # one number in three places, which is the mistake this project has
    # had to correct three times.
    #
    # Same finger width as the n device of the same flavour (1 um on lv,
    # 2 um on hv). A pmos is weaker, so the sizer answers with MORE
    # fingers, not a wider one -- and an identical wf lets the finger and
    # edge laws carry the same measured constants across both tables.
    "lv_p": dict(
        device="sg13_lv_pmos",
        core="@n.xm1.nsg13_lv_pmos",
        lib=f"{MODELS}/cornerMOSlv.lib",
        vmax="1.65", vstep="0.0125", edge_vstep="0.05",
        vsb_step=0.05,
        w="5u", ng="5",                        # 1 um fingers, as lv
        pol=-1,
        # PROVISIONAL at the short end. The subckt defaults are l=0.34u
        # for the n device and l=0.28u for the p one, but a default is
        # not a minimum: the lv nmos table sweeps from 0.13 um and
        # verifies against ngspice at 0.0000 %. pmos_probe.py measures
        # whether 0.13 um is sound here before a sweep is spent on it.
        l_spec=[(0.13, 0), (0.30, 0.01), (0.60, 0.025), (1.0, 0.05),
                (2.0, 0.1), (3.0, 0.2), (6.0, 0.5), (10.0, 1.0)],
    ),
    "hv_p": dict(
        device="sg13_hv_pmos",
        core="@n.xm1.nsg13_hv_pmos",
        lib=f"{MODELS}/cornerMOShv.lib",
        vmax="3.3", vstep="0.025", edge_vstep="0.1",
        vsb_step=0.1,
        w="10u", ng="5",                       # 2 um fingers, as hv
        pol=-1,
        # 0.40 and not the n device's 0.45: the process spec characterises
        # HV-PMOS at WxL = 10 x 0.4 um2 (VTPHV10x04) and HV-NMOS at
        # 10 x 0.45 (VTNHV10x045). The two devices do not share a shortest
        # gate and there is no reason they should.
        l_spec=[(0.40, 0), (1.0, 0.05), (2.0, 0.1), (3.0, 0.2),
                (6.0, 0.5), (10.0, 1.0)],
    ),
}

# The n-type partner of each p-type entry, so a probe can put the two in
# one netlist without either name being typed a second time.
PARTNER = {"lv_p": "lv", "hv_p": "hv"}

SYNTHETIC = dict(device="fake_nmos", core="@n.xm1.nfake_nmos",
                 lib="(synthetic: fake_model.py, no library)", pol=1)

TAG = {"mos_tt": "tt", "mos_ss": "ss", "mos_ff": "ff",
       "mos_sf": "sf", "mos_fs": "fs"}
# The corner may be given either as the library SECTION name (mos_tt)
# or as the short TAG that names the file (tt). Both are accepted and
# normalised to the section name here, so 'make_sweep hv tt' and
# 'make_sweep hv mos_tt' do the same thing. Returns None for anything
# that is neither, which the guard turns into a clear refusal.
_SHORT = {v: k for k, v in TAG.items()}


def canon_corner(c):
    if c in TAG:
        return c
    return _SHORT.get(c)


# =============================================================================
# TEMPERATURE -- a discrete selector, exactly like the corner
# =============================================================================
# '.options TEMP=27' was a literal in six netlist generators: this file,
# make_edge, make_verify, make_probe, make_mc and corner_probe. One
# physical condition in six places, which is the mistake pol and dW and
# rg each taught separately.
#
# It is a SELECTOR and not an interpolation axis, because that is what
# every documented tool does -- one complete LUT per (corner, temperature)
# pair, with the corner count C entering the memory formula as a straight
# multiplicative factor. Corners deserve that treatment on physics: a
# corner is ~17 dimensionless multipliers on PSP parameters and 'half way
# to ss' means nothing. Temperature does NOT: device behaviour is smooth
# and near-linear from -40 to 125 C, so interpolating along T alone is
# sound and is probably what ADT does, since matching a bandgap's
# temperature coefficient to under 1 ppm is not reachable from five
# isolated points.
#
# The important consequence: the DATA is the same either way. Build the
# tables indexed by temperature now and interpolation along T can be added
# later, in the engine, WITHOUT re-simulating anything. So nothing here
# forecloses that.
TEMPS = (-40, 0, 27, 85, 125)
TEMP_NOM = 27


def temp_tag(t):
    """
    The suffix a table at temperature t carries.

    TEMP_NOM carries NONE, so the 24 tables already built keep the names
    they have and every path, report and finger_laws.json that refers to
    them keeps working. A minus sign is written 'm' because a directory
    called lut_lv_p_tt_-40 is a command-line argument that starts with a
    dash.
    """
    t = int(round(float(t)))
    if t == TEMP_NOM:
        return ""
    return f"_m{-t}" if t < 0 else f"_{t}"


# =============================================================================
# POLARITY -- every netlist generator in this project imports these three
# =============================================================================
# A comment in build() below once claimed that FLAVOUR['pol'] was 'the ONLY
# place in this project that knows the device's type'. That was wrong within
# the hour: make_edge.py is a separate file, never saw pol, swept a p device
# with its gate ABOVE the source, and produced 2079 planes of a transistor
# that was off at every point. PSP normalises by NEGATING, so an applied
# VGS of +1.65 came back as vgs = -1.65 -- and pack_lut's cycle check caught
# it on plot 0, which is the only reason it did not become a table.
#
# make_verify.py, make_probe.py, make_mc.py, corner_probe.py and
# check_probe.py all write netlists too. The fix is not to repeat the sign
# in six files; it is for all six to ask here.
#
# The rule, in one place:
#   an n device wants G and D ABOVE the source and the well BELOW it
#   a p device wants G and D BELOW the source and the well ABOVE it
# and the model reports the magnitudes either way, so only the APPLIED
# voltages ever carry a sign.
def polarity(F):
    """+1 for an n-type flavour entry, -1 for a p-type one."""
    return int(F.get("pol", 1))


def applied(F, v):
    """
    The terminal voltage to APPLY at the gate or the drain for a wanted
    magnitude v. Pass the magnitude the table's axis holds; this returns
    what the source in the netlist must be set to.
    """
    return polarity(F) * float(v)


def body(F, vsb):
    """
    The body node voltage for a wanted |VSB|, with the source at 0.

    Reverse bias means the well is pushed AWAY from the source in the
    direction that widens the depletion region: below it for an n device,
    above it for a p device. So this is the opposite sign to applied().
    """
    return -polarity(F) * float(vsb)


def flavour_of(device):
    """
    Which FLAVOUR entry a STORED device name belongs to.

    Every script that writes a netlist to check a table against the
    model reads this from the table's own meta.json rather than holding
    its own copy of the device name. A verification netlist that
    simulated a different transistor from the one in the table would
    agree with nothing, or -- worse, because it is quiet -- would
    disagree by an amount small enough to be read as interpolation
    error.
    """
    for k, F in FLAVOUR.items():
        if F["device"] == device:
            return k, F
    # The synthetic device of fake_model.py / make_fake_lut.py. It is
    # NOT in FLAVOUR, so no generator offers to sweep it, but the
    # checking scripts must still recognise a table built from it:
    # when this lookup was introduced the offline test bed (make_verify
    # -> fake_spice -> check_verify on lut_fake_tt) stopped running,
    # and nothing noticed, because nothing ran it.
    if device == SYNTHETIC["device"]:
        return "fake", SYNTHETIC
    raise SystemExit(
        f"meta.json says device {device}, which is not in FLAVOUR "
        f"({sorted(f['device'] for f in FLAVOUR.values())}). Add it "
        f"there first; nothing downstream should guess.")


def build(corner, flav, temp=TEMP_NOM):
    F = FLAVOUR[flav]
    CORE, DEVICE = F["core"], F["device"]
    L_LIST = [f"{v:g}u" for v in _grid(F["l_spec"])]
    VSB_LIST = [f"{round(F['vsb_step']*i, 6):g}" for i in range(33)]
    raw = f"lut_{flav}_{TAG.get(corner, corner)}{temp_tag(temp)}.raw"
    o = []
    A = o.append
    A(f"* ADT-clone LUT sweep : {DEVICE}  {corner}")
    A(f"* standalone : ngspice -b sweep_{flav}_{corner}.spice")
    A(f"* output     : {raw}")
    A(f"* grid       : {len(L_LIST)} lengths x {len(VSB_LIST)} body biases,")
    A(f"*              VGS and VDS 0 to {F['vmax']} V in {F['vstep']} V")
    A("")
    A(f".lib {F['lib']} {corner}")
    A("")
    A(f".param w  = {F['w']}")
    A(".param l  = 1u")
    A(f".param ng = {F['ng']}")
    A("")
    # From the argument, not a literal. SG13G2's 3.3 V HV rating is
    # stated 'Maximum @ 27 C' in the process spec, so a table built at
    # 125 C holds bias points whose rating is not the one the SOA clip
    # was drawn from. That is recorded, not silently assumed away.
    A(f".options TEMP={temp:g}")
    A("")
    # From polarity() above, not from F.get here: this file is one of six
    # that write netlists and the sign must not be re-derived in each of
    # them. What comes back from the model is normalised either way, so
    # nothing downstream of the rawfile sees this at all.
    pol = polarity(F)
    sgn = "" if pol > 0 else "-"
    A(f"* S is the reference, B is driven {'negative' if pol > 0 else 'POSITIVE'}, so")
    A(f"*   VGS = {sgn}v(g)   VDS = {sgn}v(d)   VSB = {'-' if pol > 0 else ''}v(b)")
    A("* and VGS, VDS can both go into one two-source dc sweep.")
    if pol < 0:
        A("* p type: every applied voltage is negated, and PSP reports the")
        A("* magnitudes back (type = -1 in the model card), so the rawfile")
        A("* this writes has the same signs as an n-type one.")
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
    # +1 pushes the well BELOW the source, -1 ABOVE it. Same |VSB|, and
    # the model reports it the same way, so VSB_LIST is unchanged.
    #
    # NO leading '+' on the p-type branch. ngspice's preprocessor rejects
    # '+0.05' with 'PPerror: syntax error in line segment' and SKIPS the
    # alter, while everything around it carries on: 'let vhi = 1.65 - $Sv'
    # still evaluates, so the plot count still shrinks with VSB exactly as
    # it should, the sweep still reports DONE, and pack_lut still passes
    # every plot. The body simply never moves, and the finished table
    # labels 2079 planes 0 to 1.6 V that are all at VSB = 0.
    #
    # It cost a full sweep to find, it is one character, and the '+' was
    # never needed: Sv is already positive.
    A(f"    alter Vb dc = {'-' if pol > 0 else ''}$Sv")
    A("    op")
    A(f"    let vhi = {F['vmax']} - $Sv")
    # vhi is a magnitude and stays positive; the SWEEP runs toward it or
    # away from it. The SOA clip is identical either way, because VGB and
    # VDB are magnitudes too.
    if pol > 0:
        A(f"    dc Vgs 0 $&vhi {F['vstep']}  Vds 0 $&vhi {F['vstep']}")
    else:
        A("    let vlo = -$&vhi")
        A(f"    dc Vgs 0 $&vlo -{F['vstep']}  Vds 0 $&vlo -{F['vstep']}")
    A("    set filetype = binary")
    A("    set appendwrite")
    A(f"    write {raw}")
    A("    destroy all")
    A("  end")
    A("  echo ---------------- one L finished ----------------")
    A("end")
    A("")
    A("echo")
    A(f"echo ====== SWEEP DONE : {corner} ======")
    A(f"echo   wrote {raw}")
    A("echo")
    A("")
    A("quit")
    A(".endc")
    A("")
    A(".end")
    return "\n".join(o) + "\n"


def parse_temp(argv):
    """
    Pull --temp off a generator's argv, leaving the rest untouched.

    Shared by every generator so the flag means one thing everywhere and
    an unknown --flag is never silently swallowed as a positional, which
    is how pack_lut once took '--edge' for an output directory name.
    """
    rest, t = [], TEMP_NOM
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--temp":
            if i + 1 >= len(argv):
                sys.exit("--temp needs a value in C, e.g. --temp 125")
            t = float(argv[i + 1])
            i += 2
        elif a.startswith("--temp="):
            t = float(a.split("=", 1)[1])
            i += 1
        else:
            rest.append(a)
            i += 1
    if t not in [float(x) for x in TEMPS]:
        print(f"  NOTE: {t:g} C is not on the documented grid "
              f"{', '.join(str(x) for x in TEMPS)}. Building it anyway; "
              f"nothing requires the grid, but a table off it will have "
              f"no siblings to interpolate along T with later.")
    return rest, t


if __name__ == "__main__":
    args, temp = parse_temp(sys.argv[1:])
    if len(args) < 2 or args[0] not in FLAVOUR:
        sys.exit("usage: python3 make_sweep.py <flavour> <corner> ... "
                 "[--temp C]\n"
                 "   e.g. python3 make_sweep.py lv mos_tt\n"
                 "        python3 make_sweep.py lv_p mos_tt --temp 125\n"
                 f"   flavours: {', '.join(FLAVOUR)}\n"
                 f"   temperatures: {', '.join(str(x) for x in TEMPS)}"
                 f"   (default {TEMP_NOM})")
    flav = args[0]
    tt = temp_tag(temp)
    # An unknown corner used to be written straight into the .lib line
    # (TAG.get(corner, corner) fell back to whatever was typed), so
    # 'tt' in place of 'mos_tt' produced a complete-looking netlist
    # naming a library section that does not exist.
    corners = [canon_corner(c) for c in args[1:]]
    unknown = [c for c, cc in zip(args[1:], corners) if cc is None]
    if unknown:
        sys.exit(f"unknown corner {', '.join(unknown)}\n"
                 f"  give a section ({', '.join(TAG)}) "
                 f"or its short tag ({', '.join(_SHORT)})")
    for corner in corners:
        txt = build(corner, flav, temp)
        assert '"' not in txt, "double quote would break an xschem paste"
        fn = f"sweep_{flav}_{corner}{tt}.spice"
        with open(fn, "w") as fh:
            fh.write(txt)
        print(f"wrote {fn}   ->  "
              f"lut_{flav}_{TAG.get(corner, corner)}{tt}.raw"
              f"   at {temp:g} C")
    print()
    print(f"run it with:   ngspice -b sweep_{flav}_<corner>{tt}.spice")
