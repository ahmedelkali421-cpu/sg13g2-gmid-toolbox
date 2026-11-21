#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | THE CURSOR
=============================================================================
    python3 gmid_cursor.py lut_nmos_hv_tt_27 --x GM/ID=14 --y GM/GDS=80
    python3 gmid_cursor.py lut_nmos_hv_tt_27 --x GM/ID=14 --L 0.7
    python3 gmid_cursor.py lut_nmos_hv_tt_27 --x VGS=0.7  --L 0.7
    python3 gmid_cursor.py lut_hv_p_tt       --x VSTAR=0.15 --y FT=2e8 --vds 1.2
    python3 gmid_cursor.py lut_nmos_hv_tt_27 --x GM/ID=14 --y GM/GDS=80 --exact

A design chart draws a handful of curves, one per length. The point a
designer actually wants almost never sits on one of them: it sits in the
white space BETWEEN the L = 0.5 um and the L = 1 um curve. This answers
the question the chart cannot: what device IS that point, and what is
every one of its parameters there.

A point on a chart is two numbers, and at a fixed VDS and VSB the table
has exactly two free axes left, L and VGS. So the point determines the
device:

    --x NAME=VALUE --y NAME=VALUE    a point on the NAME-vs-NAME chart;
                                     L and VGS are both SOLVED
    --x NAME=VALUE --L um            the length is given, VGS is solved
    --x VGS=v      --L um            nothing to solve, a direct read

TWO MODES: read the stored table, or re-simulate the exact point:

  * default: NO NEW SIMULATION. Every number comes out of the stored
    table, through the same eval() and the same contour solver the
    charts are drawn with, so the cursor and the curve it sits beside
    cannot disagree. Instant.
  * --exact: the table still FINDS the device, and then ngspice is run
    once, behind the scenes, on exactly that device at exactly that
    bias. The card gains two columns: what the simulator says, and how
    far the table was from it. Interpolation for speed while exploring,
    the simulator for truth at the point you commit to.

--exact is the one place in this project where Python starts a
simulation by itself, and it is opt-in for that reason. The netlist it
runs is left on disk as cursor_exact.spice, a complete standalone file,
so what was simulated can be read, re-run by hand, or pasted into an
xschem code symbol. It contains no double quotes.

THREE THINGS IT REFUSES TO DO SILENTLY
  * MORE THAN ONE DEVICE. gm/gds against L is not monotone on every
    device (the HV NMOS has a reverse short-channel hump), so one chart
    point can be met at two lengths. Every solution is printed. None is
    picked for you.
  * NO DEVICE. A point above the longest curve or below the shortest is
    outside what this transistor can do. It says so, and says what range
    of y IS reachable at that x, instead of returning the nearest edge.
  * THE ACCURACY OF THE ANSWER. A solved length is almost never a stored
    node, and off-node lengths are where this table's error lives. The
    card quotes the measured interpolation error at that length.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from gmid_lut import DERIVED, ENV, corner_of_table, tokens
from gmid_size import Sizer

EXACT_NET = "cursor_exact.spice"
EXACT_OUT = "cursor_exact.txt"

# What the card shows, in the order a designer reads it. (name, unit,
# scale to display units). Anything the table cannot evaluate -- an older
# table without the noise parameters, say -- is skipped, not faked.
CARD = [
    ("bias and inversion level", [
        ("VGS", "V", 1), ("VTH", "V", 1), ("VOV", "V", 1),
        ("VDSAT", "V", 1), ("GM/ID", "S/A", 1), ("VSTAR", "mV", 1e3)]),
    ("per metre of EFFECTIVE width", [
        ("ID/W", "A/m", 1), ("GM/W", "S/m", 1)]),
    ("small signal", [
        ("GM/GDS", "", 1), ("VA", "V", 1), ("GMB/GM", "", 1),
        ("FT", "GHz", 1e-9), ("GM/ID*FT", "GHz/V", 1e-9)]),
    ("this device  (W and ng as printed above)", [
        ("ID", "uA", 1e6), ("GM", "mS", 1e3), ("GDS", "uS", 1e6),
        ("RO", "kohm", 1e-3), ("CGG_TOT", "fF", 1e15),
        ("CGS", "fF", 1e15), ("CGD", "fF", 1e15),
        ("CDTOT", "fF", 1e15), ("RG", "ohm", 1)]),
    ("noise", [
        ("GAMMA", "", 1), ("FKNEE", "kHz", 1e-3),
        ("SVG", "(nV)^2/Hz", 1e18)]),
]


def parse_pair(txt, flag):
    if "=" not in txt:
        sys.exit(f"{flag} takes NAME=VALUE, e.g. {flag} GM/ID=14")
    nm, _, v = txt.partition("=")
    try:
        return nm.strip(), float(v)
    except ValueError:
        sys.exit(f"{flag} {txt}: {v!r} is not a number")


class Cursor:
    def __init__(self, lutdir, ng=None, siz=None):
        # an existing Sizer may be handed in, so the chart tool and the
        # cursor on it read ONE loaded table and not two copies of it
        self.siz = siz if siz is not None else Sizer(lutdir)
        self.lut = self.siz.lut
        self.ng = int(ng) if ng else int(self.lut.ng)
        # whole reference fingers, exactly as gmid_xplore draws its curves
        self.W = self.ng * (self.lut.W_nom / self.lut.ng)

    # ---- VGS on the x = xval contour, for many L at once --------------
    def vgs_on(self, x, Ls, VDS, VSB):
        """(VGS, nroots) at each L. NaN where the x value is unreachable."""
        name, val = x
        Ls = np.atleast_1d(np.asarray(Ls, float))
        if name.upper() == "VGS":
            return np.full(Ls.shape, float(val)), np.ones(Ls.shape, int)
        vg = self.siz._vgs_curve(
            lambda Lq, Vq: self.lut.eval(name, L=Lq, VGS=Vq, VDS=VDS,
                                         VSB=VSB, W=self.W,
                                         ng=self.ng) / val,
            Ls, VDS, VSB)
        return vg, np.array(self.siz._roots, int)

    def y_on(self, y, x, Ls, VDS, VSB):
        """y along the x contour: the chart's own vertical slice."""
        Ls = np.atleast_1d(np.asarray(Ls, float))
        vg, _ = self.vgs_on(x, Ls, VDS, VSB)
        ok = np.isfinite(vg)
        yv = np.asarray(self.lut.eval(
            y, L=Ls, VGS=np.where(ok, vg, 0.0), VDS=VDS, VSB=VSB,
            W=self.W, ng=self.ng), float).ravel()
        return np.where(ok, yv, np.nan), vg

    # ---- the inverse: a chart point -> every device that sits on it ---
    def locate(self, x, y, VDS, VSB, sub=4, iters=40):
        """
        Every (L, VGS) with x(L,VGS) = xval and y(L,VGS) = yval.

        One dimension at a time, on purpose. Along the x contour VGS is
        a function of L, so y becomes a function of L alone, and the
        problem is a 1-D root search that can be SCANNED: every sign
        change of y(L) - yval is one solution, and all of them are
        found. A 2-D Newton iteration would return whichever one its
        starting guess fell towards and say nothing about the rest.

        The scan grid is the stored lengths plus `sub` log-spaced points
        inside each cell, so a pair of roots closer together than one
        stored cell is still separated.

        Returns (roots, (ymin, ymax)) -- the reachable range of y at
        this x, so a miss can say what WOULD have been reachable.
        """
        yname, yval = y
        g = np.asarray(self.lut.L, float)
        t = np.arange(sub + 1) / (sub + 1.0)
        Ld = np.unique(np.concatenate(
            [np.exp(np.log(g[:-1, None])
                    + t[None, :] * np.log(g[1:, None] / g[:-1, None])
                    ).ravel(), g[-1:]]))
        yv, _ = self.y_on(yname, x, Ld, VDS, VSB)
        fin = np.isfinite(yv)
        span = ((float(np.nanmin(yv)), float(np.nanmax(yv)))
                if fin.any() else (np.nan, np.nan))
        # WHERE the two ends of that range are, so that a miss can show
        # the closest device this transistor does offer.
        self.span_L = ((float(Ld[np.nanargmin(yv)]),
                        float(Ld[np.nanargmax(yv)]))
                       if fin.any() else (np.nan, np.nan))
        f = yv / yval - 1.0

        roots = []
        for i in range(len(Ld) - 1):
            if not (fin[i] and fin[i + 1]):
                continue
            if f[i] == 0.0:
                roots.append(Ld[i])
                continue
            if f[i] * f[i + 1] >= 0:
                continue
            a, b, fa = Ld[i], Ld[i + 1], f[i]
            for _ in range(iters):
                m = np.sqrt(a * b)              # bisect in log L
                fm = float(self.y_on(yname, x, [m], VDS, VSB)[0][0]
                           / yval - 1.0)
                if not np.isfinite(fm):
                    break
                if fa * fm <= 0:
                    b = m
                else:
                    a, fa = m, fm
            roots.append(np.sqrt(a * b))
        if fin[-1] and f[-1] == 0.0:
            roots.append(Ld[-1])

        out = []
        for Lr in roots:
            vg, nr = self.vgs_on(x, [Lr], VDS, VSB)
            if np.isfinite(vg[0]):
                out.append((float(Lr), float(vg[0]), int(nr[0])))
        return out, span

    # ---- the card -----------------------------------------------------
    def card(self, L, VGS, VDS, VSB, spice=None):
        """
        [(section, [(name, value, unit, spice_value_or_None)])] at one
        device. `spice` is what spice_at() returned, when --exact.
        """
        q = dict(L=L, VGS=VGS, VDS=VDS, VSB=VSB, W=self.W, ng=self.ng)
        out = []
        for title, items in CARD:
            rows = []
            for nm, unit, k in items:
                try:
                    v = float(np.asarray(self.lut.eval(nm, **q)).ravel()[0])
                except Exception:                            # noqa: BLE001
                    continue          # this table does not hold it
                if np.isfinite(v):
                    sv = None if spice is None else self.spice_value(
                        nm, spice)
                    rows.append((nm, v * k, unit,
                                 None if sv is None else sv * k))
            if rows:
                out.append((title, rows))
        return out

    # ---- the simulator-exact mode -------------------------------------
    def spice_at(self, L, VGS, VDS, VSB):
        """
        Run ngspice ONCE on this exact device at this exact bias and
        return {stored parameter name: value}, or (None, why).

        Everything that decides WHICH device is simulated is read from
        the table's own meta.json -- the device, the model library, the
        corner section, the temperature -- and the bias goes through
        make_sweep's applied()/body(), the single place that knows
        whether this is an n or a p device. Nothing here holds its own
        copy of any of it; every one of those was, at some point in this
        project, a second copy that drifted.
        """
        from make_sweep import applied, body, flavour_of
        lut = self.lut
        _, F = flavour_of(lut.device)
        try:
            corner, _src = corner_of_table(lut.meta, self.siz.dir)
        except ValueError as exc:
            return None, str(exc)
        core = F["core"]
        cols = ["vgs", "vds", "vsb"] + list(lut.params)
        o = []
        A = o.append
        A(f"* SG13G2 gm/ID Toolbox cursor, simulator-exact point : {corner}")
        A(f"* table {self.siz.dir}")
        A(f"* standalone: ngspice -b {EXACT_NET}")
        A("")
        A(f".lib {F['lib']} {corner}")
        A("")
        A(f".param w  = {self.W:.6e}")
        A(f".param l  = {L:.9e}")
        A(f".param ng = {self.ng}")
        A("")
        A(f".options TEMP={lut.temp_c:g}")
        A("")
        A("* S is the reference. Signs are already those of this device")
        A("* type: gate and drain on the conducting side of the source,")
        A("* the well on the reverse-biased side.")
        A(f"Vgs g 0 dc {applied(F, VGS):.9f}")
        A(f"Vds d 0 dc {applied(F, VDS):.9f}")
        A("Vs  s 0 dc 0")
        A(f"Vb  b 0 dc {body(F, VSB) + 0.0:.9f}")
        A(f"XM1 d g s b {F['device']} w={{w}} l={{l}} ng={{ng}} m=1")
        A("")
        for c in cols:
            A(f".save {core}[{c}]")
        A("")
        A(".control")
        A(f"shell rm -f {EXACT_OUT}")
        A("op")
        for c in cols:
            A(f"let p_{c} = {core}[{c}]")
        A("set wr_singlescale")
        A(f"wrdata {EXACT_OUT} " + " ".join("p_" + c for c in cols))
        A("quit")
        A(".endc")
        A("")
        A(".end")
        with open(EXACT_NET, "w") as fh:
            fh.write("\n".join(o) + "\n")
        if os.path.exists(EXACT_OUT):
            os.remove(EXACT_OUT)     # a stale answer must not be read
        try:
            r = subprocess.run(["ngspice", "-b", EXACT_NET],
                               capture_output=True, text=True, timeout=180)
        except FileNotFoundError:
            return None, "ngspice is not on this machine's PATH"
        except subprocess.TimeoutExpired:
            return None, "ngspice did not finish within 180 s"
        if not os.path.exists(EXACT_OUT):
            tail = "\n".join((r.stdout + r.stderr).strip()
                             .splitlines()[-6:])
            return None, ("ngspice wrote no result. Its last lines:\n"
                          + tail)
        raw = np.atleast_1d(np.loadtxt(EXACT_OUT)).ravel()
        if raw.size == len(cols) + 1:        # wr_singlescale: scale first
            raw = raw[1:]
        if raw.size != len(cols):
            return None, (f"{EXACT_OUT} holds {raw.size} numbers, "
                          f"expected {len(cols)}")
        d = {c: float(v) for c, v in zip(cols, raw)}
        # Did the simulator sit where it was asked? The model reports
        # terminal voltages as magnitudes for both device types, so the
        # comparison needs no sign.
        for nm, want in (("vgs", VGS), ("vds", VDS), ("vsb", VSB)):
            if abs(d[nm] - want) > 5e-3:
                return None, (f"the simulated device sat at {nm} = "
                              f"{d[nm]:.4f} V, not the {want:.4f} V "
                              f"asked for, so its numbers are not this "
                              f"point's. See {EXACT_NET}.")
        return d, ""

    def spice_value(self, name, d):
        """
        One card quantity, computed FROM THE SIMULATOR'S raw parameters
        by the very lambda that computes it from the table's. The
        simulated device already has the card's W and finger count, so
        no width scaling and no finger correction is applied -- which is
        the point: this column owes the table nothing.
        """
        d = dict(d)
        d["kt4"] = 4.0 * 1.380649e-23 * self.lut.temp_k
        weff = self.lut.weff(self.W, self.ng)

        def one(key):
            k = key.upper()
            if k == "VGS":
                return d["vgs"]
            if k in ("ID/W", "GM/W"):
                return DERIVED[k][0](d) / weff
            if k in DERIVED:
                return DERIVED[k][0](d)
            raise KeyError(key)

        try:
            if name.upper() in DERIVED or name.upper() == "VGS":
                return float(one(name))
            env, safe = dict(ENV), name
            for i, t in enumerate(tokens(name)):
                try:
                    env[f"_v{i}"] = one(t)
                except KeyError:
                    continue
                safe = safe.replace(t, f"_v{i}")
            return float(eval(safe, {"__builtins__": {}}, env))  # noqa: S307
        except Exception:                                    # noqa: BLE001
            return None

    def nearest(self, x, y, VDS, VSB, span):
        """
        After a miss: the device at the end of the reachable range that
        the requested y lies beyond. (L, VGS, y there), or None.

        This is NOT a solution and is never returned as one. A click a
        hair above the longest curve asked for more than the transistor
        has; the useful answer is 'the most it has is THIS', said in
        those words, not a silent snap onto the curve.
        """
        if not np.isfinite(span[0]):
            return None
        hi = y[1] > span[1]
        L = self.span_L[1] if hi else self.span_L[0]
        vg, _ = self.vgs_on(x, [L], VDS, VSB)
        if not np.isfinite(vg[0]):
            return None
        return float(L), float(vg[0]), (span[1] if hi else span[0])

    def on_node(self, L):
        g = np.asarray(self.lut.L, float)
        return bool(np.min(np.abs(g - L)) <= 1e-12 + 1e-9 * L)


def show(cur, L, VGS, VDS, VSB, nroots=1, exact=False):
    lut = cur.lut
    spice, why = (cur.spice_at(L, VGS, VDS, VSB) if exact
                  else (None, ""))
    print(f"  L   = {L * 1e6:.4f} um"
          + ("   (a stored length)" if cur.on_node(L) else
             "   (between stored lengths)"))
    print(f"  VGS = {VGS:.4f} V    VDS = {VDS:g} V    VSB = {VSB:g} V")
    print(f"  W   = {cur.W * 1e6:g} um drawn, ng = {cur.ng}"
          f"    T = {lut.temp_c:g} C")
    print("  all voltages and currents are MAGNITUDES, as the table "
          "stores them")
    if nroots > 1:
        print(f"  note: the x value is met at {nroots} gate voltages at "
              f"this length; this is the")
        print("        highest, the one carrying current. The other sits "
              "in deep weak inversion.")
    if exact and spice is None:
        print("\n  SIMULATOR-EXACT MODE FAILED, so the card below is the "
              "table alone:")
        for ln in why.splitlines():
            print("    " + ln)
    worst = (0.0, "")
    if spice is not None:
        print(f"\n     {'':<10} {'table':>13}  {'ngspice':>13}  "
              f"{'table off by':>13}")
    for title, rows in cur.card(L, VGS, VDS, VSB, spice):
        print(f"\n  -- {title}")
        for nm, v, unit, sv in rows:
            if sv is None:
                print(f"     {nm:<10} {v:>13.5g}  "
                      + ("" if spice is None else f"{'--':>13}  "
                         f"{'':>13}  ") + unit)
                continue
            e = (v / sv - 1.0) * 100.0 if sv != 0 else float("nan")
            if np.isfinite(e) and abs(e) > worst[0]:
                worst = (abs(e), nm)
            print(f"     {nm:<10} {v:>13.5g}  {sv:>13.5g}  "
                  f"{e:>+11.4f} %  {unit}")
    print("\n  -- how far to trust it")
    if spice is not None:
        print(f"     SIMULATED at this exact point: the table's worst "
              f"miss on this card is")
        print(f"     {worst[0]:.4g} %, on {worst[1]}. The ngspice column "
              f"is the truth; netlist in")
        print(f"     {EXACT_NET}.")
        return
    if cur.on_node(L):
        print("     a stored length: the L axis adds no interpolation "
              "error here")
    acc = cur.siz.l_accuracy_at(L)
    if acc is None:
        print("     no accuracy figure on file for this table; run "
              "check_probe.py")
    elif not cur.on_node(L):
        print(f"     interpolation error at this length: up to "
              f"{acc[0]:.3g} %")
        print(f"     ({acc[1]})")


def show_nearest(cur, x, y, VDS, VSB, span):
    """the closest device after a miss, labelled as NOT the answer"""
    nb = cur.nearest(x, y, VDS, VSB, span)
    if nb is None:
        return
    L, vg, yb = nb
    off = (y[1] / yb - 1.0) * 100.0
    print(f"\n  THE CLOSEST IT GETS  (not the point asked for: "
          f"{y[0]} = {yb:.5g},")
    print(f"  the request is {abs(off):.3g} % "
          f"{'above' if off > 0 else 'below'} it)")
    print("  " + "-" * 60)
    show(cur, L, vg, VDS, VSB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lutdir")
    ap.add_argument("--x", required=True,
                    help="NAME=VALUE: the point's horizontal coordinate, "
                         "e.g. GM/ID=14, VSTAR=0.15 or VGS=0.7")
    ap.add_argument("--y", default=None,
                    help="NAME=VALUE: the point's vertical coordinate; "
                         "L is then solved")
    ap.add_argument("--L", type=float, default=None,
                    help="the length in um, instead of --y")
    ap.add_argument("--vds", type=float, default=0.9)
    ap.add_argument("--vsb", type=float, default=0.0)
    ap.add_argument("--ng", type=int, default=None)
    ap.add_argument("--exact", action="store_true",
                    help="after the table finds the device, run ngspice "
                         "once on it and show table / simulator / "
                         "difference")
    a = ap.parse_args()
    if (a.y is None) == (a.L is None):
        ap.error("give exactly one of --y NAME=VALUE or --L um: a chart "
                 "point needs two coordinates, no more and no fewer")

    x = parse_pair(a.x, "--x")
    cur = Cursor(a.lutdir, ng=a.ng)
    g = np.asarray(cur.lut.L, float)

    print("=" * 72)
    print(f"  CURSOR   {a.lutdir}   ({cur.lut.device})")
    print("=" * 72)

    if a.L is not None:
        L = a.L * 1e-6
        if not (g[0] * (1 - 1e-9) <= L <= g[-1] * (1 + 1e-9)):
            sys.exit(f"  L = {a.L:g} um is outside the table, which holds "
                     f"{g[0] * 1e6:g} to {g[-1] * 1e6:g} um. Nothing is "
                     f"extrapolated.")
        vg, nr = cur.vgs_on(x, [L], a.vds, a.vsb)
        if not np.isfinite(vg[0]):
            sys.exit(f"  {x[0]} = {x[1]:g} cannot be reached at "
                     f"L = {a.L:g} um, VDS = {a.vds:g}, VSB = {a.vsb:g}: "
                     f"no gate voltage in the table gives it.")
        show(cur, L, float(vg[0]), a.vds, a.vsb, int(nr[0]), a.exact)
        return 0

    y = parse_pair(a.y, "--y")
    if y[0].upper() in DERIVED and DERIVED[y[0].upper()][1] != 0:
        sys.exit(f"  {y[0]} scales with the device width, so a chart of it "
                 f"has no meaning without a W and neither has a point on "
                 f"one. Use a width-independent quantity: {y[0]}/W, "
                 f"GM/GDS, FT, ...")
    sols, span = cur.locate(x, y, a.vds, a.vsb)
    print(f"  the point   {x[0]} = {x[1]:g} ,  {y[0]} = {y[1]:g}")
    if not sols:
        print("\n  NO DEVICE sits on this point.")
        if np.isfinite(span[0]):
            print(f"  at {x[0]} = {x[1]:g} this transistor reaches "
                  f"{y[0]} from {span[0]:.5g} to {span[1]:.5g}")
            print(f"  across its whole length range "
                  f"({g[0] * 1e6:g} to {g[-1] * 1e6:g} um); "
                  f"{y[1]:g} is outside that.")
            show_nearest(cur, x, y, a.vds, a.vsb, span)
        else:
            print(f"  {x[0]} = {x[1]:g} itself is unreachable at every "
                  f"stored length.")
        return 1
    if len(sols) > 1:
        print(f"\n  {len(sols)} DIFFERENT DEVICES sit on this point. All "
              f"are shown; choosing is yours.")
    for k, (L, vg, nr) in enumerate(sols):
        print(f"\n  ---- solution {k + 1} of {len(sols)} " + "-" * 44)
        show(cur, L, vg, a.vds, a.vsb, nr, a.exact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
