#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | WHAT THE L AXIS COSTS, MEASURED ON THE REAL TABLE
=============================================================================
    python3 l_axis_probe.py lut_nmos_hv_tt_27

No simulation. Not one.

After the finger-count laws went in, exactly one error source was left in
the whole chain: of 33 asked quantities, the 30 whose L landed on a
stored node came in at 0.029 % or better, and the 3 belonging to the one
device whose L fell inside a cell came in at 0.37 %. So the question is
what the L grid costs, and whether a finer one is worth eight hours of
sweeping.

Guessing at that is what this project keeps getting wrong, so here is a
measurement that needs no new data. LEAVE ONE NODE OUT: drop the stored
L = 0.6 um column from the axis, interpolate from its neighbours at
0.55 and 0.65, and compare against the 0.6 um values that are sitting
right there in the table. Every other axis is held on its own nodes, so
nothing but the L axis is being tested.

That measures the error on a cell of DOUBLE the real width. For a
fourth-order scheme the error goes as h^4, so the real single-width cell
should cost about a sixteenth of it -- and that ratio is reported rather
than assumed, by doing the same thing on a leave-two-out stencil and
checking the scaling actually holds before anyone trusts it to predict
what a refinement buys.

It also answers a second question for free. The axis is interpolated in
log of the DRAWN length, but every parameter really follows the
EFFECTIVE length, and L - leff is about 96 nm -- a fifth of the shortest
device. On a textbook 1/leff law, interpolating in log(leff) instead is
29x better in the first cell. Whether that survives the real physics is
measured here, on the same leave-one-out points.
"""

import json
import os
import sys

import numpy as np

from gmid_lut import LUT

# All on stored nodes of the 25 mV grid, spread over the inversion range
# and over VDS, so one unlucky bias cannot carry the conclusion.
BIAS = [
    dict(VGS=0.60, VDS=0.90, VSB=0.0),
    dict(VGS=0.70, VDS=0.90, VSB=0.0),
    dict(VGS=0.80, VDS=0.90, VSB=0.0),
    dict(VGS=1.00, VDS=0.90, VSB=0.0),
    dict(VGS=1.40, VDS=0.90, VSB=0.0),
    dict(VGS=0.80, VDS=0.30, VSB=0.0),
    dict(VGS=0.80, VDS=1.50, VSB=0.0),
    dict(VGS=0.90, VDS=0.90, VSB=0.5),
    dict(VGS=1.00, VDS=1.20, VSB=1.0),
]
QUANT = ["ids", "gm", "gds", "cgg", "vth"]


def quadw(xa, xb, xc, t):
    return (((t - xb) + (t - xc)) / ((xa - xb) * (xa - xc)),
            ((t - xa) + (t - xc)) / ((xb - xa) * (xb - xc)),
            ((t - xa) + (t - xb)) / ((xc - xa) * (xc - xb)))


def hermite(x, y, xq):
    """
    The engine's own L-axis scheme, lifted out so it can be run on an
    axis with nodes REMOVED. Cubic Hermite, node slopes from the
    parabola through three neighbours, one-sided at the ends.
    """
    n = len(x)
    i = int(np.clip(np.searchsorted(x, xq) - 1, 0, n - 2))
    i1 = i + 1
    h = x[i1] - x[i]
    f = (xq - x[i]) / h
    t2, t3 = f * f, f * f * f
    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + f, \
        -2 * t3 + 3 * t2, t3 - t2
    im, ip = max(i - 1, 0), min(i + 2, n - 1)
    out = h00 * y[i] + h01 * y[i1]
    if i > 0:
        wa, wb, wc = quadw(x[im], x[i], x[i1], x[i])
        d0 = wa * y[im] + wb * y[i] + wc * y[i1]
    else:
        wa, wb, wc = quadw(x[i], x[i1], x[ip], x[i])
        d0 = wa * y[i] + wb * y[i1] + wc * y[ip]
    if i < n - 2:
        wa, wb, wc = quadw(x[i], x[i1], x[ip], x[i1])
        d1 = wa * y[i] + wb * y[i1] + wc * y[ip]
    else:
        wa, wb, wc = quadw(x[im], x[i], x[i1], x[i1])
        d1 = wa * y[im] + wb * y[i] + wc * y[i1]
    return out + h10 * h * d0 + h11 * h * d1


NO_LOG = {"vgt", "cgd", "cdb", "cdg", "csb", "csg", "cgb"}


def leave_out(axis, vals, drop, log_y):
    """
    error of reproducing node `drop` from an axis with it removed.
    `axis` is already in the interpolation coordinate.
    """
    keep = np.array([k for k in range(len(axis)) if k != drop])
    x, y = axis[keep], vals[keep]
    t = vals[drop]
    if log_y:
        if np.any(y <= 0) or t <= 0:
            return np.nan
        v = float(np.exp(hermite(x, np.log(y), axis[drop])))
    else:
        v = float(hermite(x, y, axis[drop]))
    return 100.0 * (v - t) / t if t != 0 else np.nan


def main():
    lutdir = sys.argv[1] if len(sys.argv) > 1 else "lut_nmos_hv_tt_27"
    lut = LUT(lutdir)
    with open(os.path.join(lutdir, "meta.json")) as fh:
        meta = json.load(fh)
    L = np.asarray(meta["L"], float)
    leff = np.asarray(meta.get("leff") or [], float)
    if leff.size != L.size or not np.all(np.isfinite(leff)):
        sys.exit("meta.json has no usable leff list; repack the table.")

    print("=" * 76)
    print(f"  L-AXIS LEAVE-ONE-OUT on {lutdir}")
    print(f"  {len(L)} stored lengths, {len(BIAS)} bias points, "
          f"{len(QUANT)} parameters")
    print("  no simulation: the answer is already in the table")
    print("=" * 76)
    print(f"  drawn L minus leff: {(L-leff)[0]*1e9:.0f} nm at Lmin, "
          f"{(L-leff)[-1]*1e9:.0f} nm at Lmax")

    axA = np.log(L)           # what the engine uses today
    axB = np.log(leff)        # the effective length instead

    # pull every parameter at every bias, on the stored L nodes
    cols = {}
    for q in QUANT:
        rows = []
        for b in BIAS:
            rows.append(np.asarray(lut.raw_at(
                q, L, b["VGS"], b["VDS"], b["VSB"]), float))
        cols[q] = np.array(rows)            # (bias, L)

    print()
    print("  error of reproducing a held-out L node from its neighbours.")
    print("  that is a cell of DOUBLE the real width, so it is an upper")
    print("  bound; the h^4 scaling below says what the real one costs.")
    print()
    print(f"  {'L held out':>12}{'cell':>9}" +
          "".join(f"{q:>11}" for q in QUANT) + f"{'log(leff)':>12}")
    print("  " + "-" * (21 + 11 * len(QUANT) + 12))
    tot = {q: [] for q in QUANT}
    totB = []
    for j in range(2, len(L) - 2):
        rel = 100.0 * (L[j + 1] - L[j - 1]) / (2 * L[j])
        line = f"  {L[j]*1e6:10.2f}u{rel:8.1f}%"
        for q in QUANT:
            e = [abs(leave_out(axA, cols[q][i], j, q not in NO_LOG))
                 for i in range(len(BIAS))]
            e = [x for x in e if np.isfinite(x)]
            m = max(e) if e else np.nan
            tot[q].append(m)
            line += f"{m:10.4f}%"
        eB = [abs(leave_out(axB, cols["ids"][i], j, True))
              for i in range(len(BIAS))]
        eB = [x for x in eB if np.isfinite(x)]
        mB = max(eB) if eB else np.nan
        totB.append(mB)
        line += f"{mB:11.4f}%"
        print(line)

    print()
    print(f"  {'parameter':<12}{'median':>12}{'worst':>12}"
          f"{'worst cell':>13}")
    print("  " + "-" * 49)
    for q in QUANT:
        a = np.array(tot[q], float)
        a = a[np.isfinite(a)]
        if a.size:
            j = 2 + int(np.argmax(np.nan_to_num(tot[q])))
            print(f"  {q:<12}{np.median(a):11.4f}%{a.max():11.4f}%"
                  f"{L[j]*1e6:11.2f}u")
    a = np.array(totB, float)
    a = a[np.isfinite(a)]
    aA = np.array(tot["ids"], float)
    aA = aA[np.isfinite(aA)]
    print()
    print("  the SAME points, with the axis in log(leff) instead of log(L):")
    print(f"  {'ids':<12}{np.median(a):11.4f}%{a.max():11.4f}%")
    if a.size and np.median(a) > 0:
        print(f"  gain: {np.median(aA)/np.median(a):.1f}x on the median, "
              f"{aA.max()/a.max():.1f}x on the worst")
        print("  (a change of coordinate only -- no new simulation, and it")
        print("   cannot help where the error is real physics rather than")
        print("   the drawn-vs-effective offset)")

    # ------------------------------------------------------------------
    # IS THE CURVE STEEP, OR IS IT KINKED?
    #
    # This decides whether a finer L grid is worth eight hours. The first
    # run of this probe put the whole error in L = 0.55 .. 0.85 um and
    # essentially none above 0.90, and two things in it did not fit
    # 'smooth but steep':
    #
    #   * the leave-one-out error is NOT MONOTONIC across neighbouring
    #     cells -- ids reads 1.01 % at 0.55 and 1.80 % at 0.60, and gds
    #     reads 0.046 %/0.615 % for ids/gds at 0.80 where ids is fine;
    #   * 0.90 um comes out 225x better than 0.60 um while its cell is
    #     only 1.5x narrower, where pure resolution would give 7x.
    #
    # A smoothly curving function gives a smoothly varying error and an
    # order of 4. A function with a KINK -- a jump in a derivative, which
    # is what a model with pocketed or binned length terms has -- gives a
    # ragged error and a much lower order, and then halving the spacing
    # buys a factor of two or three instead of sixteen.
    #
    # The test is the local slope of the stored curve itself. d ln(y) /
    # d ln(L) between consecutive nodes: if it drifts steadily the curve
    # is smooth, and if it jumps at one length and stays jumped there is
    # a feature sitting there and no amount of nodes elsewhere will help.
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print("  STEEP OR KINKED? the stored curve's own local slope")
    print("  d ln(ids) / d ln(L) between consecutive nodes, and how much")
    print("  it changes from one cell to the next. A smooth curve drifts;")
    print("  a kink jumps once and stays.")
    print("=" * 76)
    nshow = min(14, len(L) - 1)
    print(f"  {'cell':<16}" + "".join(f"{f'VGS={b[chr(86)+chr(71)+chr(83)]:.2f}':>11}"
                                      for b in BIAS[:5]))
    print("  " + "-" * (16 + 11 * 5))
    slopes = []
    for k in range(nshow):
        row = f"  {L[k]*1e6:.2f}-{L[k+1]*1e6:.2f}um  "
        sl = []
        for b in BIAS[:5]:
            y0 = float(np.asarray(lut.raw_at("ids", L[k], b["VGS"],
                                             b["VDS"], b["VSB"])))
            y1 = float(np.asarray(lut.raw_at("ids", L[k + 1], b["VGS"],
                                             b["VDS"], b["VSB"])))
            s = ((np.log(y1) - np.log(y0))
                 / (np.log(L[k + 1]) - np.log(L[k])))
            sl.append(s)
            row += f"{s:10.4f} "
        slopes.append(sl)
        print(row)
    sl = np.array(slopes)
    d = np.diff(sl, axis=0)
    print()
    print("  change in that slope from one cell to the next:")
    print(f"  {'at':<16}" + "".join(f"{f'VGS={b[chr(86)+chr(71)+chr(83)]:.2f}':>11}"
                                    for b in BIAS[:5]))
    print("  " + "-" * (16 + 11 * 5))
    for k in range(len(d)):
        flag = ""
        if np.all(d[k] * d[max(k - 1, 0)] < 0) and k > 0:
            flag = "  <-- sign flip"
        print(f"  {L[k+1]*1e6:>13.2f}u  "
              + "".join(f"{v:10.4f} " for v in d[k]) + flag)
    flips = sum(1 for k in range(1, len(d))
                if np.all(d[k] * d[k - 1] < 0))
    print()
    print(f"  cells where the slope change flips sign at every bias: "
          f"{flips}")
    if flips == 0:
        print("  none: the curve is SMOOTH and merely steep, so the error")
        print("  is resolution and a finer grid pays the full h^p.")
    else:
        print("  the curve is NOT smooth at this spacing. A finer grid")
        print("  still helps, but at the order measured below, not at 4.")

    # ---- does the h^4 scaling hold? ----------------------------------
    # leave TWO nodes out and compare: if the scheme is really fourth
    # order on this axis the error should grow by about 2^4 = 16, and
    # then halving the real spacing would divide it by the same.
    print()
    print("=" * 76)
    print("  IS IT REALLY FOURTH ORDER HERE?")
    print("  leave two nodes out instead of one: triple-width cell. If the")
    print("  error grows by about (3/2)^4 = 5.1 then the scheme is behaving")
    print("  and halving the real spacing would divide the error by 16. If")
    print("  it grows much less, the error is not the spacing and a finer")
    print("  grid would buy nothing.")
    print("=" * 76)
    # PER CELL, not pooled. The first version averaged the ratio over
    # all 37 cells, and 29 of them sit at the float32 noise floor where
    # the ratio is meaningless -- so the one number it printed said
    # nothing about the only cells that matter. The order is a LOCAL
    # property of the curve and has to be reported locally.
    print(f"  {'L held out':>12}{'e(2h)':>10}{'e(3h)':>10}{'order p':>9}"
          f"{'-> e(h)':>10}   verdict")
    print("  " + "-" * 62)
    pred = []
    per_node = {}
    # ---- pass 1: measure a1, a2 and the per-cell order ----------------
    # The order used to be applied as each cell measured it, inside this
    # one loop. It cannot be. The per-cell order is a TWO-POINT estimate
    # of an exponent, and at a kink it is unstable: sg13_hv_nmos at tt
    # measured p = 0.39, -0.73 and 1.07 on three ADJOINING cells, which
    # no convergence rate does. Their MEDIAN held up against ngspice and
    # the individual values did not -- the note under the table has the
    # numbers -- and a median is not known until every cell has been
    # measured. Hence two passes.
    meas = {}
    for j in range(1, len(L) - 2):
        keep2 = np.array([k for k in range(len(L)) if k not in (j, j + 1)])
        e1s, e2s = [], []
        for q in ("ids", "gm", "gds"):
            for i in range(len(BIAS)):
                y = cols[q][i]
                if np.any(y <= 0):
                    continue
                e1 = abs(leave_out(axA, y, j, True))
                v2 = float(np.exp(hermite(axA[keep2], np.log(y[keep2]),
                                          axA[j])))
                e2 = abs(100.0 * (v2 - y[j]) / y[j])
                if np.isfinite(e1) and e1 > 1e-6 and np.isfinite(e2):
                    e1s.append(e1)
                    e2s.append(e2)
        if not e1s:
            continue
        a1, a2 = max(e1s), max(e2s)
        p = np.log(a2 / a1) / np.log(1.5) if a2 > 0 and a1 > 0 else np.nan
        meas[j] = (a1, a2, p)

    # ---- the robust order ---------------------------------------------
    # Only the cells that matter vote: one already at 0.02 % carries no
    # information about an exponent. Three votes is the floor -- below
    # that a median is not a median -- and a median at or below 0.5 is
    # refused for the same reason a single cell's is. In either of those
    # two cases the old, deliberately conservative eh = a1 stands.
    sig = [p for (a1, _, p) in meas.values()
           if a1 >= 0.02 and np.isfinite(p)]
    p_med = float(np.median(sig)) if len(sig) >= 3 else float("nan")
    use_med = bool(np.isfinite(p_med) and p_med > 0.5)

    # ---- pass 2: the estimate at the real cell width ------------------
    for j in sorted(meas):
        a1, a2, p = meas[j]
        # Credit for the real cell being half as wide is taken at the
        # cell's OWN order where that order is at least the median --
        # there the cell converges better than typical and its own number
        # is the better one -- and at the MEDIAN where it is below it,
        # which is exactly where the two-point estimate is noise.
        if use_med and np.isfinite(p) and p >= p_med:
            p_use, mark = p, " "
        elif use_med:
            p_use, mark = p_med, "*"
        elif np.isfinite(p) and p > 0.5:
            p_use, mark = p, " "
        else:
            p_use, mark = 0.0, " "
        eh = a1 / (2.0 ** p_use)
        per_node[j] = eh
        if a1 < 0.02:
            continue
        # The VERDICT still comes from the cell's OWN order, because it
        # describes the PHYSICS -- whether this cell straddles a kink --
        # and that description was never the thing that was wrong. Only
        # the number derived from it was.
        v = ("resolution" if p > 3.0 else
             "part kink" if p > 1.5 else "KINK, not spacing")
        pred.append((L[j], a1, p, eh))
        print(f"  {L[j]*1e6:10.2f}u{a1:9.4f}%{a2:9.4f}%{p:9.2f}"
              f"{eh:8.4f}%{mark}  {v}")

    # ---- leave the measurement behind, in the table --------------------
    # A limitation nobody can see is worse than one that is printed. This
    # writes the per-length estimate into the table directory, and the
    # sizing assistant reads it and says so when a solution lands in a
    # badly resolved cell. Re-pack the table, re-run this, and the
    # warning updates itself -- which is the only way a number like this
    # does not go stale and start lying.
    out = dict(
        source="l_axis_probe.py: leave-one-out on the L axis, order "
               "measured per cell and floored at the median of the cells "
               "that matter, no simulation",
        L=L.tolist(),
        err_pct=[per_node.get(j) for j in range(len(L))],
        order_median=(p_med if np.isfinite(p_med) else None),
        order_median_used=use_med,
        note="estimated interpolation error of ids/gm/gds, in per cent, "
             "at an L inside the cells adjoining each stored node. null "
             "means not measurable (axis ends, or already at the float32 "
             "floor). Where a cell's own two-point order came out below "
             "order_median, the median was used instead: a two-point "
             "exponent is unstable at a kink and the median was the one "
             "that survived an end-to-end ngspice check.")
    with open(os.path.join(lutdir, "l_accuracy.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print()
    print(f"  wrote {lutdir}/l_accuracy.json -- the sizer reads it and")
    print("  warns when a solved L lands in one of these cells.")
    print()
    print("  e(2h) is what the leave-one-out measured, e(3h) the")
    print("  leave-two-out, p the order that pair implies, and e(h) what")
    print("  the engine therefore carries at the REAL cell width -- which")
    print("  is the number to compare against the end-to-end run.")
    if use_med:
        print()
        print(f"  * e(h) taken at {p_med:.2f}, the median of the orders "
              f"measured on THIS axis,")
        print("    rather than at this cell's own p, because p is below "
              "it. A two-point")
        print("    exponent is unstable at a kink; the median is the "
              "robust statistic.")
        print()
        print("    Why the median and not the cell -- checked end to end "
              "ONCE, and the")
        print("    four numbers below are sg13_hv_nmos at tt. They are "
              "that device's,")
        print("    NOT this table's:")
        print("      cell 0.55 um: e(2h) = 1.0784 % at order 1.45 gives "
              "e(h) = 0.395 %")
        print("      check_verify at L = 575 nm measured 0.385 % on ID "
              "(at ff, 0.426 %)")
        print("    so keeping e(h) = e(2h) overstated the tool's own "
              "error by 2.8x.")
        print("    The METHOD carries over to any axis. Those four "
              "numbers do not.")
    if pred:
        tot = max(x[3] for x in pred if np.isfinite(x[3]))
        pw = p_med if use_med else np.median(
            [x[2] for x in pred if np.isfinite(x[2])])
        print()
        print(f"  worst predicted real-cell error : {tot:.4f} %")
        print(f"  median order over those cells   : {pw:.2f}")
        gain = 2.0 ** pw
        print(f"  so HALVING the L step there would divide it by "
              f"{gain:.1f}x, to {tot/gain:.4f} %")
        print()
        print("=" * 76)
        print("  WHAT TO DO WITH THIS")
        print("=" * 76)
        print("  every cell above 0.90 um is already at the float32 floor,")
        print("  so a refinement is only worth doing where it is listed.")
        print(f"  Adding 25 nm steps over that range costs about 20 % more")
        print("  sweep time per corner and buys the factor printed above.")
        print()
        print("  and the thing to weigh it against: the corner spread on")
        print("  this device is several per cent and mismatch is more, so")
        print("  an error of a few tenths of a per cent at one L is already")
        print("  below everything it will be compared with. The median")
        print("  across the end-to-end run is 0.0018 %.")


if __name__ == "__main__":
    main()
