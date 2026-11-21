#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | DEVICE XPLORE
=============================================================================
    python3 gmid_xplore.py lut_nmos_hv_tt_27 --sheet --vds 0.9
    python3 gmid_xplore.py lut_nmos_hv_tt_27 --y FT --x GM/ID --vds 0.9
    python3 gmid_xplore.py lut_nmos_hv_tt_27 --y 'GM/ID*FT' --x GM/ID
    python3 gmid_xplore.py lut_nmos_hv_tt_27 --y FT --x L --at 'GM/ID=10,15,20'
    python3 gmid_xplore.py lut_nmos_hv_tt_27 --y GM/GDS --x VDS --at GM/ID=15

The sizing assistant answers a closed question: give me a device that
meets these three numbers. This answers the open one a designer asks
FIRST -- what can this transistor do at all, and at what price.

It is an EXPRESSION ENGINE, not a fixed set of charts, because a
designer needs arbitrary charts built from complex expressions of the
device parameters. So --y takes anything eval()
takes, 'GM/ID*FT' and 'SID/(4*1.38e-23*300*GM)' included, and the nine
charts of --sheet are nine calls to the same code path. Nothing is
computed in two places.

WHAT IT IS NOT: no new simulation, ever. Every number here comes out of
the stored table.

FIVE THINGS IT REFUSES TO DO SILENTLY
  * A y-axis that depends on W is meaningless without stating W, so
    width-dependent quantities must be normalised (ID/W, GM/W, CGG_TOT/W)
    or the chart is refused with the normalisation named.
  * A y-axis that depends on the FINGER COUNT -- Cgg_tot, Cdtot, fT, and
    anything built on the junctions -- is labelled with the count it was
    drawn at, because the overlaps are affine in it and the junctions
    follow its parity. An unlabelled fT chart is a chart with a hidden
    variable.
  * Curves END where the device ends. gm/ID saturates near 1/(n*Ut), so
    a high gm/ID is unreachable at every L; the SOA clip removes the
    low-gm/ID end at high VSB; and the incomplete grid has holes by
    construction. All three come back NaN and the curve stops. Nothing
    is extrapolated and no gap is bridged.
  * The table has four degrees of freedom. --vds and --vsb take two,
    which leaves L and VGS, and a chart can only sweep one and family
    another -- so one of the two must be PINNED, with --at. A chart of
    anything versus L has nothing setting VGS, and rather than pick a
    VGS quietly it says so and stops.
  * The L family defaults to STORED LENGTHS. On a stored node the L-axis
    interpolation error is exactly zero, which on this PDK is the
    difference between 0.01 % and the 1.2 % its reverse short-channel
    hump costs at an off-node length. Asking for an off-node L is
    allowed and says so.
"""

import argparse
import json
import os
import sys

import numpy as np

from gmid_lut import LUT, NG_LAW, deps_of_expr
from gmid_size import Sizer, edge_dir_for, power_of

# The nine. The first four are the classic gm/ID design charts
# (Jespers & Murmann); the next four are the ones the JKU SG13G2 notebooks
# add, all computable from parameters already stored; the last is the
# inversion-level map that tells you which gm/ID a VGS corresponds to.
SHEET = [
    ("ID/W", "A/m", "current density -- this is what sets W", True),
    ("GM/GDS", "", "intrinsic gain available", True),
    ("FT", "Hz", "unity-current-gain frequency", True),
    ("GM/ID*FT", "Hz/(S/A)", "the speed-per-power figure of merit", True),
    ("VDSAT", "V", "headroom the device needs on VDS", False),
    ("CGG_TOT/W", "F/m", "load it presents to the stage before it", True),
    # GAMMA, not the expression with 300.15 typed into it: the engine
    # divides by the table's own temperature, the literal did not.
    ("GAMMA", "", "thermal noise factor gamma", False),
    ("FKNEE", "Hz", "flicker corner", True),
    ("VGS", "V", "the gate voltage this inversion level needs", False),
]

# x-axes that need the table INVERTED (solve for VGS), versus the ones
# that are axes of the table already.
DIRECT = ("VGS", "VDS", "VSB", "L")


class Xplore:
    def __init__(self, lutdir, edge=True, ng=None):
        self.dir = lutdir
        self.siz = Sizer(lutdir, edge=edge)   # for its contour solver
        self.lut = self.siz.lut
        self.ng = int(ng) if ng else int(self.lut.ng)
        self.nmulti = 0      # points whose bias constraint had 2 roots

    # ---- what a chart is allowed to put on its y axis -----------------
    def check_y(self, y):
        """
        (ok, note). Refuses a width-dependent y, and flags one that
        depends on the finger count rather than hiding it.
        """
        try:
            p = power_of(y)
        except Exception:                                    # noqa: BLE001
            p = None          # an expression; eval will judge it
        if p not in (None, 0):
            n = {1: "divide it by W", -1: "multiply it by W"}.get(
                p, f"its W-power is {p}")
            return False, (f"{y} scales with the device width (W-power "
                           f"{p}), so a curve of it means nothing without "
                           f"a width. Plot {y}/W instead, or {n}.")
        if not self.lut.affine_sensitive(y):
            return True, ""
        # Say WHICH parameter makes it finger-dependent. "fT depends on
        # the finger count" invites the reader to wonder whether the
        # whole chart is suspect; "via cgsol, cgdol" says it is the gate
        # overlaps, which is a fact about the layout and not about the
        # table.
        d = deps_of_expr(y, self.lut.params)
        via = ("" if d is None
               else " via " + ", ".join(sorted(d & NG_LAW)))
        return True, (f"depends on the finger count{via}: drawn at "
                      f"ng = {self.ng}")

    # ---- pinning the fourth degree of freedom -------------------------
    #
    # The table has four: L, VGS, VDS, VSB. VDS and VSB come from the
    # command line, which leaves L and VGS, and a chart can only sweep
    # one thing and family another. So EVERY chart must account for
    # exactly two of {L, VGS} -- one swept or familied, one pinned --
    # and the first version of this file did not, which is why --x L
    # crashed rather than complaining: nothing was setting VGS and the
    # code reached eval() without it.
    #
    #   --x GM/ID (or any invertible expression) : VGS is SOLVED at each
    #       x, one curve per L. The classic design chart.
    #   --x VGS   : direct, one curve per L.
    #   --x L     : VGS must be pinned, so --at is required and its
    #       values become the family. "fT versus L at gm/ID = 10, 15, 20".
    #   --x VDS or VSB : one curve per L, and --at pins VGS at each
    #       sweep point -- output characteristics at constant gm/ID, or
    #       the body effect at constant inversion level.
    #
    # --at takes NAME=VALUE. NAME may be VGS, or anything invertible.
    def _vgs_for(self, Ls, at, VDS, VSB):
        """VGS at each L that satisfies the pin. NaN where unreachable."""
        name, val = at
        Ls = np.atleast_1d(np.asarray(Ls, float))
        if name.upper() == "VGS":
            return np.full(Ls.shape, float(val))
        out = self._solve(
            lambda Lq, Vq: self.lut.eval(name, L=Lq, VGS=Vq, VDS=VDS,
                                         VSB=VSB) / val, Ls, VDS, VSB)
        return out

    def _solve(self, con, Ls, VDS, VSB):
        """every bias solve goes through here, so the multi-root count
        cannot be collected in one place and missed in another"""
        out = self.siz._vgs_curve(con, Ls, VDS, VSB)
        self.nmulti += int(self.siz._multi)
        return out

    def curve(self, y, x="GM/ID", xvals=None, L=None, VDS=0.9, VSB=0.0,
              at=None, nx=61):
        """
        One curve of y against x. Returns (xs, ys), with NaN wherever the
        device cannot be biased there -- the caller must BREAK the line
        at NaN rather than join across it, because that gap is the
        feasible boundary and it is the most useful thing on the chart.
        """
        xu = x.upper()
        W = self.ng * (self.lut.W_nom / self.lut.ng)
        kw = dict(VDS=VDS, VSB=VSB, W=W, ng=self.ng)
        X = (np.asarray(xvals, float) if xvals is not None
             else self._default_x(x, VSB, nx))

        if xu == "L":
            if at is None:
                raise ValueError(
                    "--x L sweeps the length, so nothing is left to set "
                    "VGS. Pin it with --at, e.g. --at GM/ID=15 or "
                    "--at VGS=0.7.")
            vg = self._vgs_for(X, at, VDS, VSB)
            ys = np.asarray(self.lut.eval(y, L=X, VGS=vg, **kw),
                            float).ravel()
            return X, np.where(np.isfinite(vg), ys, np.nan)

        L = float(self.lut.L[len(self.lut.L) // 2] if L is None else L)

        if xu == "VGS":
            ys = self.lut.eval(y, L=np.full(X.shape, L), VGS=X, **kw)
            return X, np.asarray(ys, float).ravel()

        if xu in ("VDS", "VSB"):
            if at is None:
                raise ValueError(
                    f"--x {xu} needs VGS pinned as well, since L is the "
                    f"family. Use --at, e.g. --at GM/ID=15.")
            ys = np.full(X.shape, np.nan)
            for k, v in enumerate(X):
                d, b = (v, VSB) if xu == "VDS" else (VDS, v)
                vg = float(self._vgs_for([L], at, d, b)[0])
                if not np.isfinite(vg):
                    continue
                ys[k] = float(np.asarray(self.lut.eval(
                    y, L=L, VGS=vg, VDS=d, VSB=b, W=W,
                    ng=self.ng)).ravel()[0])
            return X, ys

        # x is a quantity, so the table must be INVERTED: find the VGS
        # where x(L, VGS) hits each target. The sizer's own solver does
        # it -- scanning the whole VGS range for a sign change instead of
        # assuming the quantity is monotonic in VGS, and returning NaN
        # where there is no root rather than a clipped endpoint.
        #
        # It solves one constraint across many L in lockstep. Here it is
        # one L and many targets, which is the same shape turned round,
        # so the target vector rides the column axis and all 61 points
        # are solved in one pass. Same solver, same answer, 60x fewer
        # table reads -- and NOT a second implementation of the
        # bisection, which is the point.
        T = np.where(np.isfinite(X) & (X != 0), X, np.nan)

        def con(Lq, Vq):
            return self.lut.eval(x, L=Lq, VGS=Vq, VDS=VDS, VSB=VSB) / T

        vg = self._solve(con, np.full(X.shape, L), VDS, VSB)
        ys = np.asarray(self.lut.eval(
            y, L=np.full(X.shape, L),
            VGS=np.where(np.isfinite(vg), vg, 0.0), **kw), float).ravel()
        return X, np.where(np.isfinite(vg), ys, np.nan)

    def _default_x(self, x, VSB, nx):
        xu = x.upper()
        if xu == "GM/ID":
            return np.linspace(3.0, 28.0, nx)
        if xu == "VSTAR":
            return np.linspace(0.07, 0.7, nx)
        if xu == "VGS":
            return np.linspace(0.0, self.siz._vgs_hi(VSB), nx)
        if xu == "VDS":
            return np.linspace(0.0, float(self.lut.meta["vmax"]) - VSB, nx)
        if xu == "VSB":
            return np.asarray(self.lut.VSB, float)
        if xu == "L":
            return np.asarray(self.lut.L, float)
        return np.linspace(1.0, 20.0, nx)

    # ---- a family of curves -------------------------------------------
    def family(self, y, x="GM/ID", Ls=None, VDS=0.9, VSB=0.0, at=None,
               nx=61):
        """
        One curve per family member, labelled. When x is L the family is
        the PIN's values, otherwise it is the lengths.

        Lengths default to a spread of STORED nodes, so that the L axis
        contributes exactly no interpolation error -- on this PDK that is
        the difference between 0.01 % and the 1.2 % its reverse
        short-channel hump costs at an off-node length.
        """
        if x.upper() == "L":
            if at is None:
                raise ValueError("--x L needs --at to pin VGS")
            name, vals = at
            out = [(float(v), *self.curve(y, x, L=None, VDS=VDS, VSB=VSB,
                                          at=(name, float(v)), nx=nx))
                   for v in np.atleast_1d(vals)]
            return out, [], f"{name}="
        if Ls is None:
            g = np.asarray(self.lut.L, float)
            Ls = g[np.unique(np.round(
                np.linspace(0, len(g) - 1, 6)).astype(int))]
        Ls = np.atleast_1d(np.asarray(Ls, float))
        off = [float(v) for v in Ls
               if np.min(np.abs(self.lut.L - v)) > 1e-12 + 1e-9 * v]
        at1 = None if at is None else (at[0],
                                       float(np.atleast_1d(at[1])[0]))
        out = [(float(Lv), *self.curve(y, x, L=Lv, VDS=VDS, VSB=VSB,
                                       at=at1, nx=nx)) for Lv in Ls]
        return out, off, "L="


# ---------------------------------------------------------------------------
def label(flab, v, head=False):
    """
    How one family member is named. In a CSV header it carries its own
    name, because a column must stand alone once the file is opened
    somewhere else; in the legend it does not, because the legend title
    already says what the family is.
    """
    if flab == "L=":
        return f"L={v*1e6:g}u" if head else f"{v*1e6:g} um"
    return f"{flab}{v:g}" if head else f"{v:g}"


def write_csv(path, y, x, fam, flab="L="):
    """every chart writes its numbers, whether or not it can be drawn"""
    cols, head = [], []
    for v, xs, ys in fam:
        cols += [xs, ys]
        head += [f"{x}@{label(flab, v, True)}",
                 f"{y}@{label(flab, v, True)}"]
    n = max(len(c) for c in cols)
    M = np.full((n, len(cols)), np.nan)
    for j, c in enumerate(cols):
        M[:len(c), j] = c
    with open(path, "w") as fh:
        fh.write(",".join(head) + "\n")
        for r in M:
            fh.write(",".join("" if not np.isfinite(v) else f"{v:.8g}"
                              for v in r) + "\n")


def draw(ax, y, x, fam, unit="", note="", logy=None, flab="L="):
    for v, xs, ys in fam:
        m = np.isfinite(ys)
        if not m.any():
            continue
        # break the line at NaN rather than joining across a gap: the
        # gap IS the result, it is where the device cannot be biased
        ax.plot(np.where(m, xs, np.nan), np.where(m, ys, np.nan),
                lw=1.4, label=label(flab, v))
    ax.set_xlabel("L [m]" if x.upper() == "L" else x)
    if x.upper() == "L":
        ax.set_xscale("log")
    ax.set_ylabel(f"{y} [{unit}]" if unit else y)
    # A log y axis is DECIDED, not guessed: if every finite value is
    # positive and they span more than a decade, a linear axis puts the
    # short-channel end on the floor and the chart stops being readable.
    #
    # And a quantity that does not vary is not plotted as though it did.
    # The table is stored float32, so a constant comes back varying in
    # the last bit; an autoscaled axis -- a log one above all -- expands
    # that 1e-7 of nothing to the full height of the panel and draws
    # convincing noise. Below a relative span of 1e-6 the quantity is
    # constant to the table's own precision, so the axis is pinned
    # around it and the panel says so rather than lying quietly.
    v = np.concatenate([ys[np.isfinite(ys)] for _, _, ys in fam]
                       or [np.array([])])
    flat = bool(v.size and (v.max() - v.min())
                <= 1e-6 * max(abs(v.max()), abs(v.min()), 1e-300))
    if flat:
        c = float(v.mean())
        ax.set_ylim(c - max(abs(c), 1.0) * 0.1,
                    c + max(abs(c), 1.0) * 0.1)
        note = ((note + "  ") if note else "") + f"[constant: {c:.6g}]"
    else:
        if logy is None:
            logy = bool(v.size and v.min() > 0 and v.max() / v.min() > 10.0)
        if logy:
            ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    if note:
        ax.set_title(note, fontsize=8)


def attach_cursor(fig, panels, xp, x, VDS, VSB, exact=False):
    """
    Make every panel answer a click with the device at that point.

    panels maps each axes to the y expression drawn on it. The solving
    is gmid_cursor's, the SAME code the command line runs -- this function
    only turns a click into its two numbers and draws the answer back.
    A second implementation of the inversion living inside the chart
    tool would be one more place for the two to drift apart.

    Returns the handler, so it can be driven without a window.
    """
    from gmid_cursor import Cursor, show, show_nearest
    cur = Cursor(xp.dir, ng=xp.ng, siz=xp.siz)
    marks = []

    def on_click(ev):
        y = panels.get(ev.inaxes)
        if y is None or ev.xdata is None or ev.ydata is None:
            return None
        # a click while the toolbar is zooming or panning is not a query
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return None
        xv, yv = float(ev.xdata), float(ev.ydata)
        print("\n" + "=" * 72)
        print(f"  CLICK   {x} = {xv:.5g} ,  {y} = {yv:.5g}")
        print("=" * 72)
        if xv == 0 or yv == 0:
            print("  a coordinate of exactly zero cannot be solved for")
            return []
        sols, span = cur.locate((x, xv), (y, yv), VDS, VSB)
        for m in marks:
            m.remove()
        marks.clear()
        if not sols:
            print("  NO DEVICE sits on this point.")
            if np.isfinite(span[0]):
                print(f"  at {x} = {xv:.5g} this transistor reaches {y} "
                      f"from {span[0]:.5g} to {span[1]:.5g}.")
                show_nearest(cur, (x, xv), (y, yv), VDS, VSB, span)
            else:
                print(f"  {x} = {xv:.5g} is unreachable at every stored "
                      f"length.")
            marks.append(ev.inaxes.plot([xv], [yv], "x", color="0.4",
                                        ms=9, mew=2)[0])
        else:
            if len(sols) > 1:
                print(f"  {len(sols)} DIFFERENT DEVICES sit on this "
                      f"point. All are shown.")
            for k, (L, vg, nr) in enumerate(sols):
                print(f"\n  ---- solution {k + 1} of {len(sols)} "
                      + "-" * 44)
                show(cur, L, vg, VDS, VSB, nr, exact)
            txt = "\n".join(f"L = {L * 1e6:.4g} um, VGS = {vg:.4g} V"
                            for L, vg, _ in sols)
            marks.append(ev.inaxes.plot([xv], [yv], "o", color="k",
                                        ms=6, mfc="none", mew=1.5)[0])
            marks.append(ev.inaxes.annotate(
                txt, (xv, yv), xytext=(8, 8),
                textcoords="offset points", fontsize=7,
                bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)))
        fig.canvas.draw_idle()
        sys.stdout.flush()
        return sols

    fig.canvas.mpl_connect("button_press_event", on_click)
    return on_click


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("lutdir")
    ap.add_argument("--y", default=None, help="any expression eval takes")
    ap.add_argument("--x", default="GM/ID")
    ap.add_argument("--vds", type=float, default=0.9)
    ap.add_argument("--vsb", type=float, default=0.0)
    ap.add_argument("--ng", type=int, default=None)
    ap.add_argument("--L", default=None,
                    help="comma-separated lengths in um; default is a "
                         "spread of STORED lengths")
    ap.add_argument("--at", default=None,
                    help="pins the degree of freedom the x axis leaves "
                         "open, as NAME=VALUE, e.g. --at GM/ID=15 or "
                         "--at VGS=0.7 . Required for --x L (where its "
                         "comma-separated values become the family), "
                         "--x VDS and --x VSB.")
    ap.add_argument("--nx", type=int, default=61)
    ap.add_argument("--sheet", action="store_true",
                    help="the nine-chart design sheet")
    ap.add_argument("--out", default="xplore")
    ap.add_argument("--cursor", action="store_true",
                    help="open the chart in a window; a click anywhere "
                         "on a panel, BETWEEN the curves included, "
                         "prints the full device at that point")
    ap.add_argument("--exact", action="store_true",
                    help="with --cursor: every click also runs ngspice "
                         "once on the device found, and the card shows "
                         "table / simulator / difference")
    a = ap.parse_args()
    if not a.sheet and not a.y:
        ap.error("give --y EXPR, or --sheet")

    at = None
    if a.at:
        if "=" not in a.at:
            ap.error("--at takes NAME=VALUE, e.g. --at GM/ID=15")
        nm, _, vs = a.at.partition("=")
        try:
            vv = [float(v) for v in vs.split(",")]
        except ValueError:
            ap.error(f"--at {a.at}: {vs!r} is not a number")
        at = (nm.strip(), vv if len(vv) > 1 else vv[0])

    xp = Xplore(a.lutdir, ng=a.ng)
    Ls = None if not a.L else np.array(
        [float(v) * 1e-6 for v in a.L.split(",")])
    os.makedirs(a.out, exist_ok=True)

    print("=" * 72)
    print(f"  DEVICE XPLORE   {a.lutdir}")
    print(f"  VDS = {a.vds:g} V   VSB = {a.vsb:g} V   ng = {xp.ng}"
          f"   W = {xp.ng * xp.lut.W_nom / xp.lut.ng * 1e6:g} um drawn")
    if at:
        print(f"  pinned     : {at[0]} = "
              + ", ".join(f"{v:g}" for v in np.atleast_1d(at[1])))
    if xp.siz.edge_dir:
        print(f"  edge table : {xp.siz.edge_dir}  (finger laws ON)")
    else:
        print("  edge table : none -- Cgg_tot, Cdtot and fT carry the")
        print("               finger error the correction removes")
    print("=" * 72)

    todo = SHEET if a.sheet else [(a.y, "", "", None)]
    done = []
    # The bias solve depends on the x axis, not on y, so every panel of a
    # sheet hits exactly the same double roots. Printing that nine times
    # says nothing the first one did not.
    multi = 0
    for y, unit, note, logy in todo:
        ok, why = xp.check_y(y)
        if not ok:
            print(f"  REFUSED  {y}\n           {why}")
            continue
        try:
            fam, off, flab = xp.family(y, a.x, Ls=Ls, VDS=a.vds,
                                       VSB=a.vsb, at=at, nx=a.nx)
        except ValueError as exc:
            sys.exit(f"\n  {exc}")
        cov = sum(int(np.isfinite(ys).sum()) for _, _, ys in fam)
        tot = sum(len(ys) for _, _, ys in fam)
        tag = ("  [" + why + "]") if why else ""
        nm, xp.nmulti = xp.nmulti, 0
        print(f"  {y:34} {cov:5d}/{tot:<5d} points reachable{tag}")
        multi = max(multi, nm)
        if off:
            print(f"      note: {len(off)} requested length(s) are NOT "
                  f"stored nodes, so the L-axis")
            print("      interpolation error is in play there; "
                  "l_accuracy.json says how much")
        safe = "".join(c if c.isalnum() else "_" for c in y)[:40]
        write_csv(os.path.join(a.out, f"{safe}_vs_"
                               + "".join(c if c.isalnum() else "_"
                                         for c in a.x) + ".csv"),
                  y, a.x, fam, flab)
        done.append((y, unit, note, logy, fam, flab))

    if multi:
        print(f"\n  note: on {multi} of the {tot} points the x value is "
              f"met at TWO gate voltages.")
        print("        gm/ID peaks at the weak/moderate boundary and "
              "falls either side of it, so")
        print("        a high target has a root below the peak and one "
              "above. Every curve here")
        print("        takes the higher VGS -- the one carrying current; "
              "the other sits in deep")
        print("        weak inversion at a current density millions of "
              "times lower.")

    if not done:
        sys.exit("\n  nothing could be plotted")

    # ---- the picture, if matplotlib is here ---------------------------
    try:
        import matplotlib
        if not a.cursor:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                 # noqa: BLE001
        print(f"\n  CSV written to {a.out}/ . No picture: {exc}")
        print("  the numbers are the deliverable; the plot is a "
              "convenience, so")
        print("  this is not a failure. pip install matplotlib for it.")
        return

    n = len(done)
    cols = 3 if n > 4 else min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.5 * rows),
                            squeeze=False)
    for k, (y, unit, note, logy, fam, flab) in enumerate(done):
        draw(axs[k // cols][k % cols], y, a.x, fam, unit, note, logy,
             flab)
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")
    axs[0][0].legend(fontsize=7, title=done[0][5].rstrip("="),
                     title_fontsize=7)
    fig.suptitle(f"{a.lutdir}   VDS={a.vds:g} V  VSB={a.vsb:g} V  "
                 f"ng={xp.ng}"
                 + (f"   {at[0]} pinned" if at else ""), fontsize=10)
    fig.tight_layout()
    png = os.path.join(a.out, "sheet.png" if a.sheet else "chart.png")
    fig.savefig(png, dpi=150)
    print(f"\n  wrote {png}  and the CSV for every panel in {a.out}/")

    if not a.cursor:
        return
    if a.x.upper() in ("L", "VDS", "VSB"):
        print(f"\n  --cursor: not on an --x {a.x} chart. There the family "
              f"or the pin already fixes")
        print("  the device at every point of a curve, so the white "
              "space between curves is")
        print("  not a device. Use it on a chart whose family is the "
              "length (--x GM/ID,")
        print("  VSTAR, VGS or any other quantity).")
        return
    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps",
                                            "cairo", "template"):
        print(f"\n  --cursor: this Python has no window to open (matplotlib "
              f"backend {matplotlib.get_backend()!r}),")
        print("  so the chart was written as a picture only. The same "
              "answer, typed instead of")
        print("  clicked:   python3 gmid_cursor.py " + a.lutdir
              + f" --x '{a.x}=..' --y 'NAME=..' --vds {a.vds:g}")
        return
    panels = {axs[k // cols][k % cols]: done[k][0] for k in range(n)}
    attach_cursor(fig, panels, xp, a.x, a.vds, a.vsb, a.exact)
    # The saved picture is sized for a page, 3.5 inches a row; nine
    # panels of that are taller than a laptop screen and the bottom row
    # opens below the taskbar. The picture is already on disk at full
    # size, so the window can be whatever fits.
    fig.set_size_inches(min(4.6 * cols, 14.0), min(3.5 * rows, 7.6))
    fig.tight_layout()
    print("\n  CURSOR ON: click anywhere on a panel. Close the window "
          "to finish.")
    try:
        plt.show()
    except KeyboardInterrupt:
        # Ctrl-C in the terminal is a normal way to leave, not a crash
        print("\n  cursor closed")


if __name__ == "__main__":
    main()
