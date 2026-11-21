#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | ACCURACY PROBE, part 2 of 2
=============================================================================
Compares the LUT against the real model at the off-grid points.

    python3 check_probe.py

Reads probe_points.json and probe_spice.txt, asks the LUT for the same
quantities, and reports the error distribution split by inversion level,
because the answer is very different in weak and strong inversion and an
overall number would hide that.

This is the empirical answer to "is the grid dense enough". It replaces
estimates with a measurement taken on this PDK, at this grid, with this
interpolation.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, ".")
import gmid_lut                                               # noqa: E402
from gmid_lut import LUT                                      # noqa: E402


def pct(a, b):
    """signed relative error of a vs b, in percent"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (a - b) / b


def report(name, err, mask=None):
    e = err if mask is None else err[mask]
    e = e[np.isfinite(e)]
    if e.size == 0:
        print(f"  {name:<12s}  no valid points")
        return
    a = np.abs(e)
    print(f"  {name:<12s}  n={e.size:4d}   "
          f"median {np.median(a):7.3f} %   "
          f"p90 {np.percentile(a, 90):7.3f} %   "
          f"max {a.max():8.3f} %")


def main():
    with open("probe_points.json") as fh:
        spec = json.load(fh)
    cols = spec["cols"]
    pts = spec["points"]
    lut = LUT(spec["lutdir"])

    # Say out loud which table was compared against which models. A probe
    # run for one corner and checked against another table would print a
    # full, plausible report, so the pairing is stated, not assumed.
    corner = spec.get("corner", "unknown")
    with open(os.path.join(spec["lutdir"], "meta.json")) as fh:
        src = str(json.load(fh).get("source", ""))
    print("=" * 72)
    print(f"  table  : {spec['lutdir']}   (packed from {src})")
    print(f"  models : {corner}")
    if spec.get("source") and spec["source"] != src:
        sys.exit(f"\n  MISMATCH: the probe was generated for {spec['source']}"
                 f" but this table was packed from {src}.\n"
                 f"  re-run make_probe.py on the table you want to check.")
    print("=" * 72)

    raw = np.loadtxt("probe_spice.txt")
    if raw.ndim == 1:
        raw = raw[None, :]
    # wr_singlescale puts the scale in column 0
    if raw.shape[1] == len(cols) + 1:
        raw = raw[:, 1:]
    elif raw.shape[1] != len(cols):
        sys.exit(f"probe_spice.txt has {raw.shape[1]} columns, "
                 f"expected {len(cols)} or {len(cols)+1}")
    if raw.shape[0] != len(pts):
        sys.exit(f"probe_spice.txt has {raw.shape[0]} rows, "
                 f"expected {len(pts)}. Did the block run to the end?")

    S = {c: raw[:, i] for i, c in enumerate(cols)}
    L = np.array([p["L"] for p in pts])
    VGS = np.array([p["VGS"] for p in pts])
    VDS = np.array([p["VDS"] for p in pts])
    VSB = np.array([p["VSB"] for p in pts])

    # ---- did ngspice actually sit where we asked? ---------------------
    dvg = np.abs(S["vgs"] - VGS)
    dvd = np.abs(S["vds"] - VDS)
    dvb = np.abs(S["vsb"] - VSB)
    print("=" * 72)
    print("bias check : the model's internal terminal voltages vs requested")
    print(f"  |dVGS| max {dvg.max()*1e3:7.3f} mV")
    print(f"  |dVDS| max {dvd.max()*1e3:7.3f} mV")
    print(f"  |dVSB| max {dvb.max()*1e3:7.3f} mV")
    print("  small non-zero values are the series resistance drop, expected")

    # ---- the comparison ----------------------------------------------
    q = dict(L=L, VGS=VGS, VDS=VDS, VSB=VSB)
    gmid_spice = S["gm"] / S["ids"]
    ctot_spice = S["cgg"] + S["cgsol"] + S["cgdol"]

    def run(cubic, slope=True):
        gmid_lut.LUT.CUBIC = cubic
        gmid_lut.LUT.SLOPE = slope
        lu = LUT(spec["lutdir"])
        return {
            "ID":      (lu.eval("ID", **q),      S["ids"]),
            "gm":      (lu.eval("GM", **q),      S["gm"]),
            "gds":     (lu.eval("GDS", **q),     S["gds"]),
            "gm/ID":   (lu.eval("GM/ID", **q),   gmid_spice),
            "gm/gds":  (lu.eval("GM/GDS", **q),  S["gm"] / S["gds"]),
            "Cgg_tot": (lu.eval("CGG_TOT", **q), ctot_spice),
            "fT":      (lu.eval("FT", **q),      S["fug"]),
            "VTH":     (lu.eval("VTH", **q),     S["vth"]),
            "Vdsat":   (lu.eval("VDSAT", **q),   S["vdss"]),
        }

    # Captured BEFORE any run(), because run() MUTATES the class. These
    # are what gmid_lut actually ships with; everything downstream that
    # claims to measure 'the table' has to use them.
    #
    # 'tests' used to be the literal run(True, slope=True) with the
    # comment 'the shipping scheme'. That was true while gmid_lut's
    # defaults were CUBIC = True and SLOPE = True, and it went silently
    # wrong the moment SLOPE was turned off: the per-cell statistic, the
    # worst-point listing, and above all the err_meas merged into
    # l_accuracy.json -- which gmid_size.py prefers over its own bound and
    # prints as THE measured accuracy -- all carried on measuring a path
    # the engine no longer takes. It reported 80 % on a table whose real
    # error is a fraction of a per cent.
    #
    # The shipping configuration lives in ONE place. This file asks it.
    _SHIP_CUBIC = gmid_lut.LUT.CUBIC
    _SHIP_SLOPE = gmid_lut.LUT.SLOPE

    t_lin = run(False, slope=False)
    t_est = run(True, slope=False)   # cubic, node slopes ESTIMATED
    t_sup = run(True, slope=True)    # cubic, node slopes SUPPLIED from gm
    # The three above exist so the comparison tables can show EVERY path
    # whatever ships. The measurement is whichever of them the engine
    # actually takes -- reused, not re-run, so that the column printed
    # and the err_meas written to disk cannot drift apart: they are the
    # same arrays, not two evaluations that happen to agree today.
    _SHIPPED = {(False, False): t_lin,
                (True, False): t_est,
                (True, True): t_sup}
    tests = _SHIPPED.get((_SHIP_CUBIC, _SHIP_SLOPE))
    if tests is None:
        # multilinear with supplied slopes: not a configuration any of
        # the three cover, so it has to be evaluated on its own.
        tests = run(_SHIP_CUBIC, slope=_SHIP_SLOPE)

    # run() leaves the class wherever it was last set, which after the
    # three passes above is NOT the shipping configuration. Nothing
    # below evaluates the LUT today, but anything added later would
    # silently take the wrong path -- the same defect this edit exists
    # to remove. Put the class back.
    gmid_lut.LUT.CUBIC, gmid_lut.LUT.SLOPE = _SHIP_CUBIC, _SHIP_SLOPE
    _SHIP_NAME = (("cubic" if _SHIP_CUBIC else "multilinear")
                  + ", slopes "
                  + ("supplied from gm" if _SHIP_SLOPE else "estimated"))

    nan = ~np.isfinite(tests["ID"][0])
    print(f"\n  points returned NaN by the LUT : {nan.sum()} / {len(pts)}")
    if nan.any():
        print("  (those fall outside the stored grid; expected only near"
              " the SOA edge)")

    # ---- linear vs cubic, measured on this PDK -----------------------
    sat0 = VDS > S["vdss"]
    design0 = sat0 & (VDS > 0.2)
    print("\n" + "=" * 72)
    print("INTERPOLATION SCHEME : multilinear  vs  what SHIPS")
    print("  p90 of |error|, analog design space (saturated, VDS > 0.2 V)")
    print("  same stored data, no extra simulation")
    print(f"  shipping scheme: {_SHIP_NAME}")
    print("=" * 72)
    print(f"  {'quantity':<10}{'linear':>12}{'SHIPS':>12}{'gain':>10}")
    print("  " + "-" * 44)
    for k in tests:
        def p90(tt):
            e = np.abs(pct(*tt))[design0]
            e = e[np.isfinite(e)]
            return np.percentile(e, 90) if e.size else np.nan
        a, b = p90(t_lin[k]), p90(tests[k])
        print(f"  {k:<10}{a:11.3f}%{b:11.3f}%{a/b:9.1f}x"
              if b > 0 else f"  {k:<10}{a:11.3f}%{b:11.3f}%{'-':>10}")

    weak = gmid_spice > 18
    mod = (gmid_spice <= 18) & (gmid_spice > 8)
    strong = gmid_spice <= 8

    # ------------------------------------------------------------------
    # SUPPLIED versus ESTIMATED NODE SLOPES, on identical SPICE points
    #
    # The exact slope of ln(ID) along VGS is gm/ID, which is tabulated,
    # so the VGS stencil can be given its slopes instead of estimating
    # them from neighbours -- and gm then comes out as the derivative of
    # the ID interpolant rather than as a second, independent one. The
    # source paper reports an order of magnitude from this on a 50 mV
    # grid, with the gain concentrated in weak inversion, where
    # d(ln ID)/dVGS = gm/ID is nearly constant and so ln ID is nearly
    # straight.
    #
    # Measured on the synthetic table the claim held at 50 mV and then
    # FLOORED: the exact scheme's ID error was 0.00736 % at 50 mV and
    # 0.00754 % at 25 mV, an order of -0.04. Halving the VGS grid buys
    # nothing once the L axis dominates. This table is on 25 mV, so this
    # section is here to say what the change is worth HERE rather than
    # in the paper -- and the inversion-level split is what tells you
    # whether the mechanism is behaving even when the total is small.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("NODE SLOPES ON THE VGS AXIS : estimated  vs  supplied from gm")
    print(f"  (neither column is necessarily what ships -- that is "
          f"{_SHIP_NAME})")
    print("  p90 of |error|, same stored data, same SPICE points, no")
    print("  extra simulation. gm is DERIVED from the ID interpolant in")
    print("  the right-hand column, so gm/ID is consistent by construction.")
    print("=" * 72)
    print(f"  {'quantity':<10}{'region':<10}{'estimated':>12}"
          f"{'supplied':>12}{'gain':>9}")
    print("  " + "-" * 53)
    for k in ("ID", "gm", "gm/ID", "fT"):
        for rname, m in (("all", design0), ("weak", design0 & weak),
                         ("moderate", design0 & mod),
                         ("strong", design0 & strong)):
            def p90m(tt, mask=m):
                e = np.abs(pct(*tt))[mask]
                e = e[np.isfinite(e)]
                return np.percentile(e, 90) if e.size else np.nan
            # est vs sup EXPLICITLY, so this table keeps
            # showing both paths whichever one ships.
            a, b = p90m(t_est[k]), p90m(t_sup[k])
            g = a / b if (b and np.isfinite(b) and b > 0) else np.nan
            print(f"  {k:<10}{rname:<10}{a:11.4f}%{b:11.4f}%"
                  + (f"{g:8.2f}x" if np.isfinite(g) else f"{'-':>9}"))
        print("  " + "-" * 53)
    print("  fT is in the table as a control: it is built from the")
    print("  capacitances, which this change does not touch, so its two")
    print("  columns should agree except through gm.")

    print("\n" + "=" * 72)
    print("INTERPOLATION ERROR, all points")
    print("=" * 72)
    for k, (a, b) in tests.items():
        report(k, pct(a, b))

    for label, msk in [("WEAK inversion  (gm/ID > 18)", weak),
                       ("MODERATE        (8 < gm/ID < 18)", mod),
                       ("STRONG          (gm/ID < 8)", strong)]:
        print("\n" + "-" * 72)
        print(f"{label}   [{msk.sum()} points]")
        print("-" * 72)
        for k, (a, b) in tests.items():
            report(k, pct(a, b), msk)

    # ---- split by region of operation ---------------------------------
    # The inversion-level split above says nothing about triode vs
    # saturation, and that turns out to be where the error really lives.
    # Vdsat comes from the model itself, so "saturated" is the device's
    # own definition, not a guessed threshold.
    sat = VDS > S["vdss"]
    deep_tri = VDS < 0.2
    print("\n" + "=" * 72)
    print("SPLIT BY REGION OF OPERATION")
    print("=" * 72)
    for label, msk in [
            (f"SATURATION  VDS > Vdsat", sat),
            (f"triode      VDS < Vdsat", ~sat),
            (f"deep triode VDS < 0.2 V", deep_tri),
            (f"VDS > 0.2 V  (any inv.)", ~deep_tri)]:
        print(f"\n  --- {label}   [{msk.sum()} points] ---")
        for k, (a, b) in tests.items():
            report(k, pct(a, b), msk)

    # ---- ERROR BY L CELL, AGAINST SPICE ------------------------------
    # What used to be here was a "which axis carries the error" table:
    # five groups, each with one axis snapped onto grid nodes, compared
    # against an all-off-grid baseline. It was confounded and it has now
    # given a wrong answer on real data, so it is gone.
    #
    # The flaw: the five groups are DIFFERENT point subsets, drawn
    # independently (197 / 96 / 102 / 100 / 96 points on the run that
    # exposed it). Snapping an axis onto nodes can only REMOVE that
    # axis's interpolation error, so every snapped row must come in at
    # or below the baseline -- and every row came in at or ABOVE it,
    # VDS-on-grid at 7.46x. That is not an axis ranking, it is the luck
    # of which points landed in which subset, and the caption "the row
    # closest to 0 names the axis to refine" turned that noise into a
    # recommendation.
    #
    # This replaces it with a statistic the same data supports: group
    # the points by WHICH STORED L CELL they fall in, and report the
    # error per cell against SPICE. No subsetting, no baseline, and it
    # answers the question the old table was reaching for -- where on
    # the L axis the error actually lives -- while independently
    # checking l_axis_probe.py, which predicts the same thing from the
    # table alone with no simulation at all. If the two disagree,
    # this one is right: it has SPICE in it.
    base = sat & (VDS > 0.2)
    with open(os.path.join(spec["lutdir"], "meta.json")) as fh:
        Lax = np.asarray(json.load(fh)["L"], float)
    cell = np.clip(np.searchsorted(Lax, L) - 1, 0, len(Lax) - 2)
    print("\n" + "=" * 72)
    print("ERROR BY L CELL   (saturated, VDS > 0.2 V)")
    print("  the probe's own points grouped by the stored L cell they")
    print("  fall in. This is the L axis measured against SPICE, and it")
    print("  is the check on l_axis_probe.py, which predicts the same")
    print("  from the table alone.")
    print("=" * 72)
    key2 = ["ID", "gm", "gds", "gm/ID", "fT"]
    print(f"  {'L cell':<18}" + "".join(f"{k:>10}" for k in key2)
          + f"{'n':>5}")
    print("  " + "-" * (18 + 10 * len(key2) + 5))
    rows = []
    for c in range(len(Lax) - 1):
        msk = base & (cell == c)
        if msk.sum() < 3:
            continue
        vals = []
        for k in key2:
            a, b = tests[k]
            e = np.abs(pct(a, b))[msk]
            e = e[np.isfinite(e)]
            vals.append(np.percentile(e, 90) if e.size else np.nan)
        rows.append((c, vals, int(msk.sum())))
    if not rows:
        print("  (too few points per cell to break down; raise N_L"
              " in make_probe.py)")
    else:
        # SELF-CALIBRATING marker. An absolute threshold was wrong here:
        # 0.2 % flags nothing on a fine table and nearly everything on a
        # coarse one, so the mark said more about the table than about
        # the cell. Flag a cell only where it stands out AGAINST THIS
        # TABLE'S own typical cell.
        wor = np.array([np.nanmax(v) for _, v, _ in rows])
        med = np.nanmedian(wor)
        for c, vals, nn in rows:
            row = f"  {Lax[c]*1e6:6.2f}-{Lax[c+1]*1e6:<5.2f}um   "
            row += "".join(f"{v:9.3f}%" for v in vals) + f"{nn:5d}"
            w = np.nanmax(vals)
            print(row + ("   <--" if np.isfinite(w) and med > 0
                         and w > 3 * med else ""))
        print()
        # ---- the measurement DISPLACES the estimate -----------------
        # l_axis_probe.py writes a per-length error map from the table
        # alone, and where its order test fails it keeps the wide-cell
        # number as a conservative bound rather than dividing by an
        # exponent the data refuses. That bound is honest and it is
        # loose: on this table it reads 0.615 % in the 0.80-0.85 um
        # cell where SPICE here says 0.014 %, a factor of 44. The sizer
        # reads that file to warn about a solution's length, so left
        # alone it raises a false alarm in exactly the number that was
        # built to be trusted.
        #
        # So this merges the MEASURED per-cell errors into the same
        # file, tagged with how many points each rests on. The sizer
        # prefers measured over estimated and says which it used. A cell
        # the probe never visited keeps the bound; one it visited with
        # three points says so, because a p90 of three points is not a
        # measurement either.
        acc_path = os.path.join(spec["lutdir"], "l_accuracy.json")
        try:
            acc = {}
            if os.path.exists(acc_path):
                with open(acc_path) as fh:
                    acc = json.load(fh)
            if list(acc.get("L", [])) != list(Lax):
                acc = {"L": Lax.tolist(),
                       "err_pct": [None] * len(Lax)}
            em = acc.get("err_meas") or [None] * len(Lax)
            nm = acc.get("n_meas") or [0] * len(Lax)

            # THIS run's numbers, accumulated on their own. The merge
            # used to take max() against whatever was already on disk,
            # which meant a value written by a broken probe could never
            # be lowered by a correct re-run -- the reason
            # strip_bad_meas.py had to exist at all, and the reason the
            # five tables whose damage fell under its 50 % threshold
            # kept it. A node this run visited now takes this run's
            # value outright; max() still applies WITHIN the run,
            # because a node bounds the two cells on either side of it
            # and a query there carries the worse of them.
            this_em = [None] * len(Lax)
            this_nm = [0] * len(Lax)
            for c, vals, nn in rows:
                w = float(np.nanmax(vals))
                if not np.isfinite(w):
                    continue
                for j in (c, c + 1):
                    if this_em[j] is None or w > this_em[j]:
                        this_em[j], this_nm[j] = w, nn
            for j in range(len(Lax)):
                if this_em[j] is not None:
                    em[j], nm[j] = this_em[j], this_nm[j]

            acc["err_meas"], acc["n_meas"] = em, nm
            # The scheme is stamped in, because the number is only
            # valid for the path it was measured on. A later flip of
            # gmid_lut.LUT.SLOPE leaves this field disagreeing with the
            # engine, and a reader can now SEE that instead of
            # inheriting it as though it still applied.
            acc["measured_by"] = ("check_probe.py, p90 of |error| vs "
                                  "SPICE per stored L cell, "
                                  f"interpolation: {_SHIP_NAME}")
            acc["measured_scheme"] = dict(cubic=bool(_SHIP_CUBIC),
                                          slope=bool(_SHIP_SLOPE))
            with open(acc_path, "w") as fh:
                json.dump(acc, fh, indent=1)
            print(f"  merged into {acc_path} as err_meas -- the sizer")
            print("  now warns from SPICE where SPICE exists, and from")
            print("  the table-only bound only where it does not.")
            print()
        except Exception as exc:                             # noqa: BLE001
            print(f"  (could not merge into l_accuracy.json: {exc})")
            print()
        print(f"  typical cell, worst quantity in it : {med:.3f} %")
        print("  cells marked <-- are more than 3x that, i.e. they stand")
        print("  out against this table rather than against a number")
        print("  someone chose. What is in them is a question for")
        print("  l_axis_probe.py, which measures the same axis from the")
        print("  table alone and says per cell whether the cause is the")
        print("  spacing or a feature of the model that spacing will not")
        print("  fix. Where the two disagree, believe this one: it has")
        print("  SPICE in it.")

    # ---- the number that actually matters for analog design ----------
    design = sat & (VDS > 0.2)
    print("\n" + "=" * 72)
    print(f"ANALOG DESIGN SPACE : saturated AND VDS > 0.2 V "
          f"[{design.sum()} points]")
    print("=" * 72)
    for k, (a, b) in tests.items():
        report(k, pct(a, b), design)

    # ---- where is the error worst? ------------------------------------
    e = np.abs(pct(tests["ID"][0], tests["ID"][1]))
    e = np.where(np.isfinite(e), e, -1)
    order = np.argsort(e)[::-1][:8]
    print("\n" + "=" * 72)
    print("worst 8 points for ID")
    print("=" * 72)
    print(f"  {'L[um]':>8} {'VGS':>7} {'VDS':>7} {'VSB':>6} "
          f"{'Vdsat':>7} {'gm/ID':>7} {'sat?':>5} {'err%':>9}")
    for i in order:
        if e[i] < 0:
            continue
        print(f"  {L[i]*1e6:8.3f} {VGS[i]:7.3f} {VDS[i]:7.3f} {VSB[i]:6.3f} "
              f"{S['vdss'][i]:7.3f} {gmid_spice[i]:7.2f} "
              f"{'yes' if sat[i] else 'NO':>5} {e[i]:9.3f}")

    print("\n" + "=" * 72)
    print("how to read this")
    print("=" * 72)
    print("  gm/ID is the number the sizing actually turns on. If its p90")
    print("  is under ~1 % the grid is dense enough for design work.")
    print("  ID carries the largest error in weak inversion, where it is")
    print("  exponential in VGS; that is the axis to refine if anything.")


if __name__ == "__main__":
    main()
