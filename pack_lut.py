#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | PACKER
Turns the multi-plot ngspice rawfile from the sweep into a 4-D LUT stored
on an incomplete grid.
=============================================================================

    python3 pack_lut.py lut_nmos_hv_tt_27.raw

Produces a directory  lut_nmos_hv_tt_27/  holding one memory-mappable
.npy per parameter plus meta.json. Memory-mapped because the table is
~1.2 GB and you do not want to load all of it to read one point.

It validates every plot before trusting it:
  * point count must equal n*n for the SOA-clipped sweep at that VSB
  * the VGS vector must cycle 0 .. vhi exactly n times
  * the measured vsb must equal the VSB that plot was supposed to run at
  * leff must be constant inside a plot and must increase with L
Anything that fails is reported with its plot index and the pack aborts,
because a silently misplaced plot would corrupt every chart downstream.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rawstream import iter_plots, short_name                    # noqa: E402

# =============================================================================
# CONFIG - must match the sweep exactly
# =============================================================================

# ---------------------------------------------------------------------------
# WHICH DEVICE THIS RAWFILE CAME FROM
#
# The grid is not stored in the rawfile -- ngspice writes plots, not
# axes -- so the packer must know the sweep that produced it in order to
# shelve each plot at the right (L, VSB) and to rebuild the incomplete
# VGS/VDS plane. Every one of these numbers has to match the sweep
# EXACTLY. A VMAX of 3.3 applied to a 1.65 V rawfile does not fail; it
# reconstructs the wrong plane and writes a table that is quietly wrong,
# so the L list and the spec both come from make_sweep.py rather than
# being retyped here.
#
# DEVICE, VMAX, VSTEP and the VSB step are read from the same FLAVOUR
# table the sweep was generated from. Pick with --flavour lv; the
# default stays hv so every existing command line keeps its meaning.
# ---------------------------------------------------------------------------
from make_sweep import (FLAVOUR as _SWEEP, grid_m as _grid_m,
                        TEMP_NOM as _TNOM, temp_tag as _ttag)        # noqa: E402

FLAVOUR_NAME = "hv"
_F = _SWEEP[FLAVOUR_NAME]

L_LIST = _grid_m(_F["l_spec"])
VSB_LIST = [round(_F["vsb_step"] * i, 6) for i in range(33)]

VSTEP = float(_F["vstep"])
VMAX = float(_F["vmax"])         # VGB and VDB ceiling used for the clip
DEVICE = _F["device"]
# Overridden by --temp; recorded into meta.json either way.
TEMP = _TNOM

# W_REF is MEASURED off the rawfile's own weff, not trusted from here.
# It used to be a hand-entered constant, which is a number that has to
# be kept in step with a simulation by hand -- and the LV device has no
# measurement until its first sweep finishes. The value below is now
# only a cross-check: if the rawfile disagrees with it, that is reported
# rather than silently overridden.
W_REF = 9.7e-6                   # measured: weff 1.94u per finger x nf 5
W_NOM = 10.0e-6                  # what was asked for in the netlist
NG = 5

# The edge-term sweep is the SAME sweep at ng = 1 on a coarser voltage
# grid, so VSTEP, NG, W_NOM and W_REF become arguments rather than
# constants. Everything else -- the L list, the VSB list, the incomplete
# grid, every validation -- is unchanged, because the only thing that
# differs is how many fingers and how finely the plane is sampled.
#   --vstep 0.1   --ng 1
# W_REF follows from ng: one finger of 1.94 um effective.
WF_EFF = W_REF / NG              # 1.94 um, the reference finger
WF_NOM = W_NOM / NG              # 2 um drawn

# Parameters to keep. Must exist in the rawfile.
PARAMS = [
    "ids", "gm", "gmb", "gds", "vth", "vgt", "vdss",
    "cgg", "cgs", "cgd", "cgb", "cdg", "cdb", "csg", "csb",
    "cgsol", "cgdol", "cjd", "cjs",
    "fug", "vearly", "beff", "rg",
    "sfl", "sid", "sig", "fknee", "cigid",
    "igs", "igd", "igb", "iavl", "ijd",
]
# read but not stored: used only for validation
CHECK = ["vgs", "vds", "vsb", "weff", "leff"]


# =============================================================================
def n_at(vsb):
    """number of VGS (= number of VDS) points for this VSB after SOA clip"""
    return int(round((VMAX - vsb) / VSTEP)) + 1


class Grid:
    """
    Incomplete (VGS, VDS, VSB) grid, flattened VGS-fastest.

        FGi = iVGS + iVDS*nG + iVSB*nG*nD          full grid
        IGi = FGi + OS[iVDS, iVSB]                 incomplete grid

    The mask here is exactly what was simulated: at a given VSB the sweep
    ran 0..vhi on both VGS and VDS, so valid = (iVGS < n) & (iVDS < n).
    No boundary padding: a point we did not simulate must read as NaN,
    not as a neighbour.
    """

    def __init__(self, vgs, vds, vsb):
        self.vgs, self.vds, self.vsb = vgs, vds, vsb
        nG, nD, nB = len(vgs), len(vds), len(vsb)
        self.shape = (nG, nD, nB)

        valid = np.zeros(self.shape, dtype=bool)
        self.n_b = np.array([n_at(v) for v in vsb])
        for b, n in enumerate(self.n_b):
            valid[:n, :n, b] = True
        self.valid = valid

        counts = valid.sum(axis=0)                       # (nD, nB)
        cumul = np.cumsum(counts.T.ravel()) - counts.T.ravel()
        start_ig = cumul.reshape(nB, nD).T
        start_fg = (np.arange(nD)[:, None] * nG
                    + np.arange(nB)[None, :] * nG * nD)
        self.OS = (start_ig - start_fg).astype(np.int64)
        self.n_valid = int(valid.sum())

    def flat_index(self, ig, idx, ib):
        nG, nD, _ = self.shape
        return ig + idx * nG + ib * nG * nD + self.OS[idx, ib]

    def slice_for(self, ib):
        """flat slice (start, stop) covering the whole VSB block ib"""
        n = self.n_b[ib]
        start = self.flat_index(0, 0, ib)
        return start, start + n * n


# =============================================================================
def main():
    global VSTEP, NG, W_REF, W_NOM, WF_EFF, WF_NOM
    global L_LIST, VSB_LIST, VMAX, DEVICE, FLAVOUR_NAME
    argv = [a for a in sys.argv[1:]]
    # --vstep and --ng, so the coarse ng=1 edge sweep goes through the
    # same validated packer instead of a near-copy of it.
    # --flavour picks the device, and with it the whole grid, from the
    # same table make_sweep.py generated the netlist from.
    global TEMP
    i = 0
    pos = []
    flav = None
    while i < len(argv):
        if argv[i] == "--vstep":
            VSTEP = float(argv[i + 1]); i += 2
        elif argv[i] == "--ng":
            NG = int(argv[i + 1]); i += 2
            W_REF, W_NOM = NG * WF_EFF, NG * WF_NOM
        elif argv[i] == "--flavour":
            flav = argv[i + 1]; i += 2
        elif argv[i] == "--temp":
            TEMP = float(argv[i + 1]); i += 2
        else:
            # An unknown --flag used to land here and become a positional,
            # which made it the OUTPUT DIRECTORY NAME. '--edge', invented
            # on the spot, nearly wrote a table into a folder called
            # '--edge'; a typo like '--flavor' would pack with the wrong
            # grid and say nothing at all.
            if argv[i].startswith("--"):
                sys.exit(f"unknown option {argv[i]}\n"
                         f"  known: --flavour --vstep --ng --temp")
            pos.append(argv[i]); i += 1
    if flav is not None:
        # set BEFORE --ng is applied below, because the finger widths it
        # multiplies are per flavour
        if flav not in _SWEEP:
            sys.exit(f"unknown flavour {flav}; have {sorted(_SWEEP)}")
        FLAVOUR_NAME = flav
        F = _SWEEP[flav]
        L_LIST = _grid_m(F["l_spec"])
        VSB_LIST = [round(F["vsb_step"] * i2, 6) for i2 in range(33)]
        VMAX, DEVICE = float(F["vmax"]), F["device"]
        if not any(a == "--vstep" for a in argv):
            VSTEP = float(F["vstep"])
        # the DRAWN finger, read as its own shortest decimal so that
        # 10u/5 is the double 2e-6 and not one ULP away from it
        WF_NOM = float(f"{float(F['w'].rstrip('u')) / int(F['ng']):g}e-6")
        WF_EFF = WF_NOM            # replaced by the measurement below
        W_NOM = float(f"{float(F['w'].rstrip('u')) * NG / int(F['ng']):g}e-6")
        W_REF = NG * WF_EFF
    if not pos:
        sys.exit("usage: python3 pack_lut.py <rawfile> [outdir] "
                 "[--flavour lv] [--vstep 0.1] [--ng 1] [--temp 125]")
    rawpath = pos[0]
    outdir = pos[1] if len(pos) > 1 else \
        os.path.splitext(os.path.basename(rawpath))[0]
    print(f"  grid step {VSTEP*1e3:g} mV   ng {NG}   "
          f"W_ref {W_REF*1e6:g}u eff / {W_NOM*1e6:g}u drawn   "
          f"TEMP {TEMP:g} C")
    # The temperature is in the raw file's NAME, put there by make_sweep,
    # and in --temp here. If the two disagree the table would be labelled
    # with a temperature it was not simulated at, which is the worst kind
    # of wrong: everything downstream would trust the label.
    _want = _ttag(TEMP)
    _base = os.path.splitext(os.path.basename(rawpath))[0]
    _seen = ""
    for _t in (-40, 0, 85, 125):
        if _base.endswith(_ttag(_t)):
            _seen = _ttag(_t)
            break
    if _seen != _want:
        sys.exit(f"  the rawfile is named {_base!r}, whose temperature tag "
                 f"is {_seen or '(none, so 27 C)'},\n"
                 f"  but --temp says {TEMP:g} C (tag "
                 f"{_want or '(none)'}). One of the two is wrong, and a\n"
                 f"  table labelled with a temperature it was not "
                 f"simulated at is worse than no\n"
                 f"  table: make_verify, make_probe and make_mc all read "
                 f"that label and trust it.")

    nL, nB = len(L_LIST), len(VSB_LIST)
    vgs_grid = np.round(np.arange(0, VMAX + 1e-9, VSTEP), 6)
    vds_grid = vgs_grid.copy()
    g = Grid(vgs_grid, vds_grid, np.array(VSB_LIST))

    print("=" * 70)
    print(f"packing {rawpath}")
    print("=" * 70)
    print(f"  L        : {nL} values  {L_LIST[0]*1e6:g}u .. {L_LIST[-1]*1e6:g}u")
    print(f"  VGS, VDS : {len(vgs_grid)} values  0 .. {VMAX} V  step {VSTEP*1e3:g} mV")
    print(f"  VSB      : {nB} values  0 .. {VSB_LIST[-1]} V")
    print(f"  expected plots      : {nL*nB}")
    print(f"  valid points per L  : {g.n_valid:,}")
    print(f"  valid points total  : {nL*g.n_valid:,}")
    full = nL * int(np.prod(g.shape))
    print(f"  full grid would be  : {full:,}   "
          f"-> {100*(1-nL*g.n_valid/full):.1f} % saved by the SOA clip")
    print(f"  parameters          : {len(PARAMS)}")
    print(f"  on disk             : "
          f"{nL*g.n_valid*len(PARAMS)*4/2**30:.2f} GiB")
    print("=" * 70)

    os.makedirs(outdir, exist_ok=True)
    store = {p: np.lib.format.open_memmap(
        os.path.join(outdir, p + ".npy"), mode="w+",
        dtype=np.float32, shape=(nL, g.n_valid)) for p in PARAMS}

    want_long = None          # filled on the first plot
    problems = []
    seen = 0
    leff_seen = np.full(nL, np.nan)
    weff_seen = np.full(nL, np.nan)

    for k, hdr, data in iter_plots(rawpath):
        if want_long is None:
            # map short names -> the long '@n.xm1...[x]' names in the file
            long_of = {}
            for v in data:
                long_of.setdefault(short_name(v), v)
            missing = [p for p in PARAMS + CHECK if p not in long_of]
            if missing:
                sys.exit(f"these are not in the rawfile: {missing}\n"
                         f"available: {sorted(long_of)}")
            want_long = long_of

        iL, iB = divmod(k, nB)
        if iL >= nL:
            problems.append(f"plot {k}: more plots than {nL}x{nB}")
            break

        vsb_want = VSB_LIST[iB]
        n = n_at(vsb_want)
        npts = len(data[want_long["ids"]])

        # ---- validation -------------------------------------------------
        if npts != n * n:
            problems.append(
                f"plot {k} (L={L_LIST[iL]*1e6:g}u VSB={vsb_want}): "
                f"{npts} points, expected {n*n}")
            continue

        vg = data[want_long["vgs"]]
        vb = data[want_long["vsb"]]
        le = data[want_long["leff"]]
        we = data[want_long["weff"]]

        # VGS must restart every n points
        if abs(vg[0]) > 0.02 or abs(vg[n - 1] - (n - 1) * VSTEP) > 0.05 \
                or (npts > n and abs(vg[n]) > 0.02):
            problems.append(
                f"plot {k}: VGS does not cycle as expected "
                f"(vgs[0]={vg[0]:.4g} vgs[{n-1}]={vg[n-1]:.4g} "
                f"vgs[{n}]={vg[n] if npts > n else float('nan'):.4g})")
            continue

        if abs(np.median(vb) - vsb_want) > 0.02:
            problems.append(
                f"plot {k}: measured vsb={np.median(vb):.4g}, "
                f"expected {vsb_want}")
            continue

        lmed = float(np.median(le))
        if np.isnan(leff_seen[iL]):
            leff_seen[iL] = lmed
        elif abs(lmed - leff_seen[iL]) > 1e-9:
            problems.append(
                f"plot {k}: leff={lmed:.6g} but earlier plots of this L "
                f"gave {leff_seen[iL]:.6g}")
            continue

        # weff is PER FINGER: the subcircuit hands the device w/ng. It
        # must not move with L or with the body bias, so recording it
        # per L and checking the spread afterwards is both the
        # measurement and its own validation.
        wmed = float(np.median(we))
        if np.isnan(weff_seen[iL]):
            weff_seen[iL] = wmed

        # ---- store --------------------------------------------------------
        lo, hi = g.slice_for(iB)
        for p in PARAMS:
            store[p][iL, lo:hi] = data[want_long[p]].astype(np.float32)

        seen += 1
        if k % 100 == 0:
            print(f"  plot {k:5d}/{nL*nB}  L={L_LIST[iL]*1e6:6.2f}u  "
                  f"VSB={vsb_want:4.2f}  {npts:6d} pts", flush=True)

    for p in PARAMS:
        store[p].flush()

    # ---- leff must grow with L -------------------------------------------
    if not np.all(np.diff(leff_seen) > 0):
        bad = np.where(np.diff(leff_seen) <= 0)[0]
        problems.append(f"leff is not increasing with L at indices {bad}; "
                        f"the L loop may not have taken effect")

    # ---- the finger width, measured rather than declared -----------------
    # Everything the table reports per unit width rests on this number.
    # Taking it from the rawfile means it cannot drift away from the
    # simulation that produced the data; the constant at the top of this
    # file is kept only to say so when the two disagree.
    wf = weff_seen[np.isfinite(weff_seen)]
    if wf.size == 0:
        problems.append("no weff was read from the rawfile; W_ref is the "
                        "declared value and may be wrong")
    else:
        spread = (wf.max() - wf.min()) / wf.mean()
        if spread > 1e-6:
            problems.append(
                f"weff moves with L by {spread*100:.4g} % "
                f"({wf.min():.6g} to {wf.max():.6g}); it should be fixed")
        declared = W_REF
        W_REF = float(np.median(wf)) * NG
        print(f"  finger   : weff {W_REF/NG*1e6:.4g}u measured x ng {NG} "
              f"-> W_ref {W_REF*1e6:.4g}u eff")
        if abs(W_REF - declared) / declared > 1e-3:
            print(f"  NOTE     : the declared W_ref was "
                  f"{declared*1e6:.4g}u; the rawfile says "
                  f"{W_REF*1e6:.4g}u, and the rawfile wins.")

    # TEMP belongs to the TABLE, not to whoever reads it. Everything
    # downstream -- make_verify, make_probe, check_probe, make_mc -- must
    # run at the temperature the table was BUILT at, because a comparison
    # across two temperatures measures the temperature and nothing else.
    # So it is recorded here, once, and read from here, never passed as a
    # flag that could disagree. A table packed before TEMP existed has no
    # key, and a reader treats that as the nominal.
    meta = dict(
        device=DEVICE, source=os.path.basename(rawpath), temp=TEMP,
        W_ref=W_REF, W_nom=W_NOM, ng=NG,
        L=L_LIST, VGS=vgs_grid.tolist(), VDS=vds_grid.tolist(),
        VSB=VSB_LIST, vstep=VSTEP, vmax=VMAX,
        params=PARAMS, n_valid=g.n_valid,
        leff=leff_seen.tolist(),
        OS=g.OS.tolist(), n_b=g.n_b.tolist(),
    )
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh)

    print("-" * 70)
    print(f"  plots stored : {seen} / {nL*nB}")
    if problems:
        print(f"  PROBLEMS     : {len(problems)}")
        for pr in problems[:25]:
            print("     " + pr)
        if len(problems) > 25:
            print(f"     ... and {len(problems)-25} more")
        print("\n  the table is INCOMPLETE. do not use it yet.")
    else:
        print("  validation   : every plot passed")
        print(f"\n  wrote {outdir}/  "
              f"({sum(os.path.getsize(os.path.join(outdir,f)) for f in os.listdir(outdir))/2**30:.2f} GiB)")
    print("-" * 70)

    print("\n  leff per L (m), should rise monotonically:")
    for i in range(0, nL, 6):
        row = "  ".join(f"{L_LIST[j]*1e6:5.2f}u:{leff_seen[j]*1e6:6.3f}u"
                        for j in range(i, min(i + 6, nL)))
        print("    " + row)


if __name__ == "__main__":
    main()
