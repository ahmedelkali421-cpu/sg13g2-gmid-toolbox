#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | END-TO-END VERIFICATION, part 2 of 2
=============================================================================
    python3 check_verify.py

Three comparisons, in order of what they actually prove:

  1. SPICE vs the SPECIFICATION
     Did the device the tool handed you meet the numbers you asked for?
     This is the only line that matters to a designer, and it is the one
     nothing before this script has tested.

  2. SPICE vs the SIZER'S PREDICTION
     The report prints a page of quantities at the solution. This asks the
     model whether any of them is wrong. A disagreement here is a table or
     interpolation error, not a sizing error.

  3. the finger studies (section 4), which decide whether the table's
     one-finger-times-ng assumption holds at all

Plus a geometry check on every device: nf and weff read back from the
model against what was intended. Total W versus W per finger was the first
silent bug in this project, and ng is now chosen automatically.
"""

import json
import os
import sys

import numpy as np


def fmt(v, unit=""):
    import math
    if v is None or not np.isfinite(v):
        return "  --  "
    if v == 0:
        return f"0 {unit}".strip()
    e = math.floor(math.log10(abs(v)) / 3) * 3
    e = max(-18, min(12, e))
    pre = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "u", -3: "m",
           0: "", 3: "k", 6: "M", 9: "G", 12: "T"}[e]
    return f"{v/10**e:.4g} {pre}{unit}".strip()


def pct(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) \
            or b == 0:
        return np.nan
    return 100.0 * (a - b) / b


def main():
    with open("verify_cases.json") as fh:
        spec = json.load(fh)
    cols, cases = spec["cols"], spec["cases"]
    raw = np.loadtxt("verify_spice.txt")
    if raw.ndim == 1:
        raw = raw[None, :]
    ncol = len(cols) + 1                      # the trailing nf column
    if raw.shape[1] == ncol + 1:              # wr_singlescale prepends scale
        raw = raw[:, 1:]
    if raw.shape[1] != ncol:
        sys.exit(f"verify_spice.txt has {raw.shape[1]} columns, "
                 f"expected {ncol} or {ncol+1}")
    if raw.shape[0] != len(cases):
        sys.exit(f"verify_spice.txt has {raw.shape[0]} rows, expected "
                 f"{len(cases)}. Did the block run to the end?")
    I = {c: i for i, c in enumerate(cols)}
    NF = len(cols)

    print("=" * 76)
    print(f"  table  : {spec['lutdir']}   models : {spec['corner']}")
    print(f"  devices: {len(cases)}")
    print("=" * 76)

    # ---- geometry, first, because everything else rests on it ---------
    print()
    print("=" * 76)
    print("GEOMETRY READ BACK FROM THE MODEL")
    print("  weff must equal W_drawn - ng*dW, and nf must equal ng.")
    print("  W total versus W per finger was the first silent bug here.")
    print("=" * 76)
    print(f"  {'#':>3} {'kind':<8}{'ng asked':>9}{'nf got':>8}"
          f"{'W drawn':>11}{'weff want':>11}{'weff got':>11}{'err':>9}")
    print("  " + "-" * 64)
    bad_geom = 0
    # dW comes from the TABLE, not from here. It used to be the literal
    # 0.06e-6, which is sg13_hv_nmos at its 2 um finger. On the lv table
    # the model's own dW is -0.02e-6 -- the effective width is WIDER
    # than drawn, not narrower -- and the literal flagged all eleven
    # devices as broken by exactly 8.511 %. One identical number across
    # eleven different geometries is never physics; it is a constant
    # that belongs to another device.
    with open(os.path.join(spec["lutdir"], "meta.json")) as fh:
        _m = json.load(fh)
    dW = (_m["W_nom"] - _m["W_ref"]) / _m["ng"]
    # A dead band, because 'narrower' and 'WIDER' are claims and a float
    # difference of 1e-22 m is not. sg13_hv_pmos measures dW = 0 exactly,
    # and the comparison dW > 0 called +3.388e-13 nm 'narrower effective
    # than drawn' -- a reading of float noise, printed as a measurement.
    # 0.1 nm is below anything the model resolves and far below the 3 nm
    # edge roughness that sets the mismatch floor.
    if abs(dW) < 1e-10:
        _sense = "effectively NO width correction"
    else:
        _sense = (("narrower" if dW > 0 else "WIDER")
                  + " effective than drawn")
    print(f"  dW from the table: {dW*1e9:+.4g} nm per finger ({_sense})")

    # ---- what the geometry studies measure, for the table to keep -----
    # gmid_size.py printed five of these as literals measured on
    # sg13_hv_nmos at its 2 um finger, and said so: 'it has NOT been
    # remeasured for this table'. Saying so was honest; measuring them is
    # better, and the studies below measure every one of the five.
    #
    # So they go into the table directory exactly the way l_axis_probe
    # leaves l_accuracy.json there: the report quotes THIS table's
    # numbers, a re-run refreshes them, and pmos will get its own without
    # anyone editing a constant. Hard-coding the lv numbers beside the hv
    # ones would have replaced five borrowed constants with ten.
    LAWS = dict(device=_m["device"], ng_ref=_m["ng"],
                wf_ref_drawn=_m["W_nom"] / _m["ng"],
                wf_ref_eff=_m["W_nom"] / _m["ng"] - dW)
    for k, c in enumerate(cases):
        if c["spec"] < 0:
            continue
        nf = raw[k, NF]
        we_want = c["W"] - c["ng"] * dW
        we_got = raw[k, I["weff"]] * nf       # per-finger weff x fingers
        e = pct(we_got, we_want)
        flag = ""
        if abs(nf - c["ng"]) > 1e-9 or not np.isfinite(e) or abs(e) > 0.5:
            flag, bad_geom = "  <-- CHECK", bad_geom + 1
        print(f"  {c['spec']:>3} {c['kind']:<8}{c['ng']:>9}{nf:>8.0f}"
              f"{fmt(c['W'],'m'):>11}{fmt(we_want,'m'):>11}"
              f"{fmt(we_got,'m'):>11}{e:>8.3f}%{flag}")
    print(f"\n  geometry problems: {bad_geom}")

    # ---- 1. did the device meet the specification? -------------------
    print()
    print("=" * 76)
    print("1. SPICE vs THE SPECIFICATION")
    print("  the only number a designer cares about")
    print("=" * 76)
    SP = {"ID": lambda r: r[I["ids"]],
          "GM": lambda r: r[I["gm"]],
          "GDS": lambda r: r[I["gds"]],
          "GM/ID": lambda r: r[I["gm"]] / r[I["ids"]],
          "GM/GDS": lambda r: r[I["gm"]] / r[I["gds"]],
          "FT": lambda r: r[I["fug"]],
          "VTH": lambda r: r[I["vth"]],
          "VDSAT": lambda r: r[I["vdss"]],
          "CGG_TOT": lambda r: r[I["cgg"]] + r[I["cgsol"]] + r[I["cgdol"]],
          "VSTAR": lambda r: 2 * r[I["ids"]] / r[I["gm"]],
          "VA": lambda r: r[I["ids"]] / r[I["gds"]],
          "CDTOT": lambda r: r[I["cdb"]] + r[I["cjd"]] + r[I["cgdol"]],
          "CGD": lambda r: r[I["cgd"]],
          "L": lambda r: np.nan}
    # The stored L axis, so the report can say whether each solution
    # landed ON a node or inside a cell. This is not decoration: on the
    # first real run every specification whose L sat on a node came in
    # at 0.024 % or better and the single one that did not came in at
    # 0.37 %, on ID and gm and gds together. Without this column that
    # reads as one bad device; with it, it reads as the L-axis
    # interpolation error, which is what it is.
    Lax = np.asarray(spec.get("L_axis") or [], float)

    def l_on_node(L):
        if Lax.size == 0:
            return None
        return bool(np.min(np.abs(Lax - L)) <= 1e-12 + 1e-9 * L)

    worst, worst_lhs = [], []
    for k, c in enumerate(cases):
        if c["kind"] != "built":
            continue
        onn = l_on_node(c["L"])
        tag = "" if onn is None else ("   L on a stored node"
                                      if onn else
                                      "   L INSIDE a cell, so the L-axis "
                                      "interpolation is in play")
        print(f"\n  spec {c['spec']}:  VDS={c['VDS']:g} VSB={c['VSB']:g}"
              f"   ->  L={fmt(c['L'],'m')} W={fmt(c['W'],'m')} "
              f"ng={c['ng']} VGS={fmt(c['VGS'],'V')}{tag}")
        for key, want in c["knobs"].items():
            ku = key.upper()
            if ku == "L":
                got, e = c["L"], pct(c["L"], want)
            elif ku in SP:
                got = float(SP[ku](raw[k]))
                e = pct(got, want)
            else:
                continue
            # What the sizer itself says the rounding of the width to a
            # whole finger did to THIS quantity: its own 'asked for'
            # column against its 'realised' one. That is right for a
            # quantity proportional to W, right for one independent of
            # it, and right for one that is neither -- Cgg_tot is affine
            # in the finger count, so it carries a FRACTION of the
            # rounding -- and it needs no list of which is which.
            #
            # The list that used to be here named five quantities by
            # hand, and it had already gone stale twice over: it charged
            # Cgg_tot with the whole rounding when the edge correction
            # means it carries part, and it let fT off entirely when fT
            # is the quantity that made the finger count matter in the
            # first place.
            a = (c.get("asked") or {}).get(ku)
            p = (c.get("pred") or {}).get(ku)
            qz = pct(p, a) if (a and p) else np.nan
            show = np.isfinite(qz) and abs(qz) > 0.005
            left = e - qz if show else e
            print(f"      {key:<9} asked {fmt(want):>12}   spice "
                  f"{fmt(got):>12}   {e:+8.3f} %"
                  + (f"   of which {qz:+.2f} % is the finger rounding,"
                     f" leaving {left:+.3f} %" if show else ""))
            if np.isfinite(e):
                worst.append((abs(e), c["spec"], key))
            if np.isfinite(left):
                worst_lhs.append((abs(left), c["spec"], key, onn))

    # ---- 2. did the table mispredict anything? -----------------------
    print()
    print("=" * 76)
    print("2. SPICE vs THE SIZER'S PREDICTION")
    print("  every quantity the report printed, against the model")
    print("=" * 76)
    keys = [k for k in spec["predict"] if k in SP]
    print(f"  {'#':>3} {'kind':<8}" + "".join(f"{k:>9}" for k in keys))
    print("  " + "-" * (11 + 9 * len(keys)))
    allerr = {k: [] for k in keys}
    for k, c in enumerate(cases):
        if c["spec"] < 0:
            continue
        row = f"  {c['spec']:>3} {c['kind']:<8}"
        for q in keys:
            p = c["pred"].get(q)
            g = float(SP[q](raw[k]))
            e = pct(g, p)
            allerr[q].append(e)
            row += f"{e:8.3f}%" if np.isfinite(e) else f"{'--':>9}"
        print(row)
    print()
    print(f"  {'quantity':<10}{'median |err|':>14}{'max |err|':>12}")
    print("  " + "-" * 36)
    for q in keys:
        e = np.abs(np.array(allerr[q], float))
        e = e[np.isfinite(e)]
        if e.size:
            print(f"  {q:<10}{np.median(e):13.4f}%{e.max():11.4f}%")

    # ---- 4. the three geometry studies -------------------------------
    def rows_of(tag):
        out = []
        for k, c in enumerate(cases):
            kd = str(c["kind"])
            if kd.startswith(tag):
                out.append((k, c, kd))
        return out

    A, B, C = rows_of("A/"), rows_of("B/"), rows_of("C/")
    bad_p = 0

    if A:
        print()
        print("=" * 76)
        print("4A. CONSTANT FINGER WIDTH, VARYING ng")
        print("  every device is the reference finger repeated, which is the")
        print("  assumption the whole table rests on. ID/ng must be constant.")
        print("=" * 76)
        print(f"  {'ng':>4}{'finger weff':>13}{'ID':>12}{'ID/ng':>13}"
              f"{'vs ng=1':>10}{'VTH':>10}{'Cgg/ng':>12}")
        print("  " + "-" * 74)
        base = None
        spread = []
        for k, c, kd in A:
            n = raw[k, NF]
            idn = raw[k, I["ids"]] / n
            cgn = (raw[k, I["cgg"]] + raw[k, I["cgsol"]]
                   + raw[k, I["cgdol"]]) / n
            if base is None:
                base = idn
            d = 100.0 * (idn - base) / base
            spread.append(d)
            print(f"  {int(n):>4}{fmt(raw[k, I['weff']],'m'):>13}"
                  f"{fmt(raw[k, I['ids']],'A'):>12}{fmt(idn,'A'):>13}"
                  f"{d:9.3f}%{fmt(raw[k, I['vth']],'V'):>10}"
                  f"{fmt(cgn,'F'):>12}")
        sp = max(spread) - min(spread)
        LAWS["id_per_ng_spread_pct"] = float(sp)
        LAWS["ng_studied"] = [int(round(raw[k, NF])) for k, c, kd in A]
        print(f"\n  spread of ID/ng : {sp:.4f} %")
        if sp < 0.1:
            print("  PROPORTIONAL. At a fixed finger width the table's")
            print("  assumption is exact, so sizing by the NUMBER of")
            print("  fingers removes the error at its root.")
        else:
            print("  NOT proportional even at a fixed finger width, so")
            print("  there is a finger-to-finger effect too and sizing by")
            print("  ng alone would not be exact either.")

        # ---- the exponents, measured the RIGHT way --------------------
        # Study C necessarily changes the finger width, so the exponents
        # it reports are the layout effect, not the parameter's scaling.
        # THIS is the exponent that matters: how a parameter scales with
        # the NUMBER of reference fingers, which is the only thing the
        # engine ever multiplies by. Fitted over the whole ng range by
        # least squares in log-log, so a single odd point cannot set it.
        try:
            sys.path.insert(0, ".")
            from gmid_lut import RAW_P
        except Exception:
            RAW_P = {}
        ngs = np.array([raw[k, NF] for k, c, kd in A], float)
        print()
        print("  exponent with respect to the FINGER COUNT, log-log fit")
        print("  over the whole range above. This is what RAW_P must hold.")
        print(f"  {'param':<9}{'at ng=1':>13}{'at ng=max':>13}"
              f"{'p vs ng':>9}{'RAW_P':>7}{'resid':>9}")
        print("  " + "-" * 60)
        wrong = 0
        for nm in cols:
            v = np.array([raw[k, I[nm]] for k, c, kd in A], float)
            if not np.all(np.isfinite(v)) or np.any(np.abs(v) < 1e-30):
                continue
            if np.all(v > 0):
                y = np.log(v)
            elif np.all(v < 0):
                y = np.log(-v)
            else:
                continue
            x = np.log(ngs)
            pm, b = np.polyfit(x, y, 1)
            res = float(np.max(np.abs(y - (pm * x + b))))
            want = RAW_P.get(nm)
            flag = ""
            # A big residual means the parameter is not a power law in ng
            # AT ALL, so the fitted exponent is meaningless and RAW_P is
            # not what is wrong. Saying "RAW_P WRONG" for those was my
            # own misreading: the residual column was already telling me.
            if res > 1e-4:
                flag = "  <-- not a power law (affine)"
            elif want is not None and abs(pm - want) > 0.01:
                flag, wrong = "  <-- RAW_P WRONG", wrong + 1
            print(f"  {nm:<9}{fmt(v[0]):>13}{fmt(v[-1]):>13}"
                  f"{pm:9.4f}{'' if want is None else want:>7}"
                  f"{res:9.2e}{flag}")
        print(f"\n  entries of RAW_P contradicted by this: {wrong}")
        print("  parameters marked affine share source/drain diffusions")
        print("  between fingers, so no exponent can express them. The")
        print("  gate overlaps are affine in the count; the JUNCTIONS")
        print("  follow its parity, because region ng+1 is a source when")
        print("  ng is even and a drain when it is odd. 4D measures both.")
        print("  'resid' is the worst departure from a straight line in")
        print("  log-log; a clean power law gives ~1e-15.")

        # ------------------------------------------------------------------
        # 4D. THE AFFINE CORRECTION ITSELF, WITH INTERPOLATION TAKEN OUT
        #
        # Study A sweeps the finger count at ONE bias, and that bias was
        # chosen on grid nodes: L = 1 um is in the stored L list, VGS =
        # 1.1 V and VDS = 0.9 V are whole multiples of both the 25 mV
        # main grid and the 100 mV edge grid, and VSB = 0. So on BOTH
        # tables every stencil weight lands on a stored point and the
        # interpolation error is not small, it is zero. Whatever this
        # section shows is the affine model and nothing else.
        #
        # It exists because a hypothesis died here. Cdtot was sitting at
        # 0.15 % median and 0.66 % max and I reasoned it was the coarse
        # 100 mV edge grid feeding e = (5*P1 - P5)/4 with a 1.25x lever.
        # That lever is ZERO at ng = 5 and grows toward 0.25 as ng grows.
        # The measured errors do the opposite -- 2.9 % at ng = 6, zero at
        # ng = 11, 83 and 128 -- so the reasoning was wrong, and the two
        # specifications added to test it (L pinned to a node, fT and
        # Cdtot as targets) came back at 0.001 % and 0.002 %. Hence this:
        # a measurement in place of an argument. The simulations for it
        # were already in the file.
        # ------------------------------------------------------------------
        try:
            from gmid_lut import LUT
            from gmid_size import edge_dir_for
            lut = LUT(spec["lutdir"])
            ed = edge_dir_for(spec["lutdir"])
            bare = LUT(spec["lutdir"])          # same table, no correction
            if ed:
                lut.attach_edge(ed)
        except Exception as exc:                 # noqa: BLE001
            lut = None
            print(f"\n  (4D skipped: {exc})")

        if lut is not None and lut.edge is not None:
            print()
            print("=" * 76)
            print("4D. THE FINGER-COUNT LAWS AGAINST SPICE")
            print("  L = 1 um, VGS = 1.1 V, VDS = 0.9 V, VSB = 0: every one")
            print("  of those is a stored node on BOTH tables, so the")
            print("  interpolation error here is zero, not small. What is")
            print("  left is the finger-count law and nothing else: affine")
            print("  for the gate overlaps, parity-dependent for the")
            print("  junctions, plain proportionality for the rest.")
            print("=" * 76)
            # all five of the affine set, then the three quantities built
            # from them, then two controls that are purely proportional
            # and must read zero in BOTH columns
            QS = [("cgsol", 1), ("cgdol", 1), ("cjd", 1), ("cjs", 1),
                  ("ijd", 1), ("CGG_TOT", 0), ("CDTOT", 0), ("FT", 0),
                  ("CGG", 0), ("ID", 0)]
            SPQ = {"CGG_TOT": lambda r: (r[I["cgg"]] + r[I["cgsol"]]
                                         + r[I["cgdol"]]),
                   "CDTOT": lambda r: (r[I["cdb"]] + r[I["cjd"]]
                                       + r[I["cgdol"]]),
                   "FT": lambda r: r[I["fug"]],
                   "CGG": lambda r: r[I["cgg"]],
                   "ID": lambda r: r[I["ids"]]}
            print(f"  {'quantity':<9}{'ng':>4}{'spice':>13}"
                  f"{'law':>11}{'proportional':>14}"
                  f"{'err law':>10}{'err prop':>11}")
            print("  " + "-" * 72)
            worst_aff = {}
            # worst_aff is the error of the CORRECTED value and must sit
            # at the float32 floor. What the report needs to quote is the
            # other column: the cost of ignoring the laws. And for the
            # junctions it must be split by PARITY, because that is the
            # law -- an even count is a different device end from an odd
            # one, and averaging the two hides the whole effect.
            worst_prop, worst_prop_even = {}, {}
            for nm, _ in QS:
                for k, c, kd in A:
                    ng = int(round(raw[k, NF]))
                    g = SPQ[nm](raw[k]) if nm in SPQ else raw[k, I[nm]]
                    q = dict(L=c["L"], VGS=c["VGS"], VDS=c["VDS"],
                             VSB=c["VSB"], W=c["W"], ng=ng)
                    try:
                        a = float(np.asarray(lut.eval(nm, **q)).ravel()[0])
                        p = float(np.asarray(bare.eval(nm, **q)).ravel()[0])
                    except Exception:            # noqa: BLE001
                        continue
                    ea, ep = pct(a, g), pct(p, g)
                    worst_aff[nm] = max(worst_aff.get(nm, 0.0),
                                        abs(ea) if np.isfinite(ea) else 0.0)
                    if np.isfinite(ep):
                        worst_prop[nm] = max(worst_prop.get(nm, 0.0),
                                             abs(ep))
                        if ng % 2 == 0:
                            worst_prop_even[nm] = max(
                                worst_prop_even.get(nm, 0.0), abs(ep))
                    print(f"  {nm:<9}{ng:>4}{fmt(g):>13}{fmt(a):>11}"
                          f"{fmt(p):>14}{ea:9.3f}%{ep:10.3f}%")
                print("  " + "-" * 72)
            print("  worst |err| of the corrected value, per quantity:")
            for nm, w in sorted(worst_aff.items(), key=lambda t: -t[1]):
                print(f"    {nm:<9}{w:9.4f} %")
            LAWS["law_exact_worst_pct"] = (max(worst_aff.values())
                                           if worst_aff else None)
            LAWS["cost_if_ignored_pct"] = {k: float(v)
                                           for k, v in worst_prop.items()}
            LAWS["cost_if_ignored_even_pct"] = {
                k: float(v) for k, v in worst_prop_even.items()}
            print()
            print("  'proportional' is the same table with the edge table")
            print("  detached: what the tool did before any of this. If")
            print("  'err aff' is at the float32 floor across the whole ng")
            print("  range, INCLUDING the even counts, then the laws are")
            print("  exact and any residual elsewhere comes from the BIAS,")
            print("  not from the finger count. The even counts are the")
            print("  test: an affine law fitted from ng = 5 and ng = 1,")
            print("  both odd, reproduces every odd count perfectly and")
            print("  is out by tens of per cent at ng = 2.")

    if B:
        print()
        print("=" * 76)
        print("4B. CONSTANT TOTAL WIDTH, VARYING FINGER WIDTH")
        print("  same channel area. Anything that moves here is a")
        print("  layout-dependent effect and nothing else.")
        print("=" * 76)
        # Compared as DENSITIES, per metre of effective width, so that
        # any residual difference in total weff divides out and what is
        # left is the layout effect alone.
        print(f"  {'ng':>4}{'finger weff':>13}{'total weff':>12}"
              f"{'ID/weff':>12}{'vs widest':>11}{'Cgg/weff':>12}"
              f"{'vs widest':>11}")
        print("  " + "-" * 74)
        b0 = None
        wf_seen, e_id, e_cg = [], [], []
        for k, c, kd in B:
            n = raw[k, NF]
            tot = raw[k, I["weff"]] * n
            cg = (raw[k, I["cgg"]] + raw[k, I["cgsol"]]
                  + raw[k, I["cgdol"]]) / tot
            jd = raw[k, I["ids"]] / tot
            if b0 is None:
                b0 = (jd, cg)
            wf_seen.append(float(raw[k, I["weff"]]))
            e_id.append(abs(float(pct(jd, b0[0]))))
            e_cg.append(abs(float(pct(cg, b0[1]))))
            print(f"  {int(n):>4}{fmt(raw[k, I['weff']],'m'):>13}"
                  f"{fmt(tot,'m'):>12}{fmt(jd,'A/m'):>12}"
                  f"{pct(jd, b0[0]):10.3f}%"
                  f"{fmt(cg,'F/m'):>12}{pct(cg, b0[1]):10.3f}%")
        # The study's narrowest finger is not necessarily the TABLE's own.
        # NG_B was chosen when the reference finger was 2 um, and on a
        # table built at 1 um the reference sits one step OUTSIDE this
        # range -- so the number is then a floor and not a measurement of
        # the effect at the width the table actually holds. Record which
        # of the two it is instead of letting the report present a floor
        # as a measurement.
        nar = min(wf_seen)
        LAWS["finger_width_effect_pct"] = max(e_id)
        LAWS["finger_width_cgg_pct"] = max(e_cg)
        LAWS["wf_narrowest_studied"] = nar
        # Both ends, because the figure above is a spread and a spread
        # without its range is not a number: '22.8 % on the current
        # density' answers nothing until it says over WHAT change of
        # finger width. With ng = 50 in NG_B the range now runs past the
        # hv reference finger, so truncating it at the reference would be
        # one more judgement call buried in a constant. Printing the two
        # ends removes the need for the judgement.
        LAWS["wf_widest_studied"] = max(wf_seen)
        LAWS["wf_ref_inside_study"] = bool(nar <= LAWS["wf_ref_eff"] * 1.02)
        print("\n  if ID moves and Cgg_tot does not, the effect is on the")
        print("  MOBILITY, not on the geometry: that is STI stress, and the")
        print("  table can only be right at the finger width it was built")
        print("  at.")
        if not LAWS["wf_ref_inside_study"]:
            print()
            print(f"  NOTE: the narrowest finger studied here is "
                  f"{fmt(nar,'m')} and this table's")
            print(f"  own reference finger is {fmt(LAWS['wf_ref_eff'],'m')} "
                  f"effective, which is NARROWER.")
            print("  The effect grows as the finger narrows, so the figure "
                  "above is a FLOOR")
            print("  at this table's width, not a measurement of it. Add "
                  "that finger count")
            print("  to NG_B in make_verify.py to measure it instead.")

    if len(C) >= 2:
        try:
            sys.path.insert(0, ".")
            from gmid_lut import RAW_P
        except Exception:
            RAW_P = {}
        a, b = C[0][0], C[-1][0]
        r_w = (raw[b, I["weff"]] * raw[b, NF]) / \
              (raw[a, I["weff"]] * raw[a, NF])
        print()
        print("=" * 76)
        print("4C. CONSTANT ng, WIDTH SCALED  (the conflated study)")
        print(f"  Weff ratio {r_w:.4f} at ng = {int(raw[a, NF])}, so the")
        print("  finger width changes by the same factor. An exponent that")
        print("  is not an integer here is that, not a broken parameter.")
        print("  Values below 1e-30 are skipped: they are solver noise.")
        print("=" * 76)
        print(f"  {'param':<9}{'narrow':>13}{'wide':>13}{'ratio':>9}"
              f"{'p meas':>8}{'RAW_P':>7}")
        print("  " + "-" * 59)
        for nm in cols:
            va, vb = raw[a, I[nm]], raw[b, I[nm]]
            if not np.isfinite(va) or not np.isfinite(vb) \
                    or abs(va) < 1e-30:
                continue
            ratio = vb / va
            pm = np.log(ratio) / np.log(r_w) if ratio > 0 else float("nan")
            want = RAW_P.get(nm)
            flag = ""
            if want is not None and np.isfinite(pm) \
                    and abs(pm - want) > 0.02:
                flag, bad_p = "  <-- not an integer", bad_p + 1
            print(f"  {nm:<9}{fmt(va):>13}{fmt(vb):>13}{ratio:9.4f}"
                  f"{pm:8.3f}{'' if want is None else want:>7}{flag}")
        print(f"\n  parameters whose exponent is not what RAW_P says: "
              f"{bad_p}")

    # ---- 0. did the device even sit where we asked? ------------------
    # The sizer returns the bias the TABLE is indexed by, which is the
    # model's INTERNAL vgs/vds/vsb. verify.spice applies those numbers to
    # the external sources. If the subckt has series resistance, the
    # internal bias is then lower by the IR drop -- and since the table
    # itself was built the same way, that drop would be applied TWICE.
    # In weak inversion ID is exponential in VGS, so 17 mV is 40 % of ID.
    # This section decides whether the misses below are physics or my
    # netlist, and it costs nothing: the numbers are already in the file.
    print()
    print("=" * 76)
    print("0. BIAS CHECK: where the device actually sat")
    print("  asked minus the model's own internal terminal voltages")
    print("=" * 76)
    print(f"  {'#':>3} {'kind':<9}{'dVGS':>10}{'dVDS':>10}{'dVSB':>10}"
          f"{'ID err':>10}")
    print("  " + "-" * 52)
    dvmax = 0.0
    for k, c in enumerate(cases):
        dg = (c["VGS"] - raw[k, I["vgs"]]) * 1e3
        dd = (c["VDS"] - raw[k, I["vds"]]) * 1e3
        db = (c["VSB"] - raw[k, I["vsb"]]) * 1e3
        ide = pct(raw[k, I["ids"]], c["pred"].get("ID")) \
            if c["pred"].get("ID") else float("nan")
        dvmax = max(dvmax, abs(dg), abs(dd))
        print(f"  {c['spec']:>3} {str(c['kind']):<9}{dg:9.3f}m{dd:9.3f}m"
              f"{db:9.3f}m{ide:9.3f}%")
    print(f"\n  largest |dVGS| or |dVDS| : {dvmax:.3f} mV")
    if dvmax > 1.0:
        print("  NOT NEGLIGIBLE. The device is not at the bias the table")
        print("  was indexed by, so the ID misses below are at least")
        print("  partly my netlist, not the model disagreeing.")
    else:
        print("  negligible, so the device sat where the table says it")
        print("  should. The misses below are real.")

    # ---- 1b. is it the width scaling? --------------------------------
    # The measured exponent of ids from the W-scaling pair predicts the
    # error that extrapolating from W_ref should cause:
    #     err = (W/W_ref)^(p-1) - 1
    # If the measured ID errors line up with that, width scaling is the
    # explanation. If they do not, it is something else and I should stop
    # blaming it.
    ws = [k for k, c in enumerate(cases)
          if str(c["kind"]).startswith("wscale")]
    p_ids = None
    if len(ws) == 2:
        a, b = ws
        r_w = (raw[b, I["weff"]] * raw[b, NF]) / \
              (raw[a, I["weff"]] * raw[a, NF])
        r_i = raw[b, I["ids"]] / raw[a, I["ids"]]
        if r_w > 0 and r_i > 0:
            p_ids = float(np.log(r_i) / np.log(r_w))
    meta_ref = None
    try:
        with open(os.path.join(spec["lutdir"], "meta.json")) as fh:
            meta_ref = json.load(fh)
    except Exception:
        pass
    if p_ids is not None and meta_ref:
        Wref = float(meta_ref["W_ref"])
        print()
        print("=" * 76)
        print("1b. IS THE WIDTH SCALING THE CULPRIT?")
        print(f"  measured exponent of ids : p = {p_ids:.4f}  (1 would be")
        print(f"  exact proportionality).  W_ref = {fmt(Wref,'m')} effective,")
        print(f"  {fmt(Wref/meta_ref['ng'],'m')} per finger at ng="
              f"{meta_ref['ng']}")
        print("=" * 76)
        print(f"  {'#':>3} {'kind':<9}{'Weff/Wref':>11}{'finger':>10}"
              f"{'ID err':>10}{'predicted':>11}{'left over':>11}")
        print("  " + "-" * 65)
        for k, c in enumerate(cases):
            if c["spec"] < 0 or not c["pred"].get("ID"):
                continue
            weff_tot = raw[k, I["weff"]] * raw[k, NF]
            ratio = weff_tot / Wref
            fing = weff_tot / raw[k, NF]
            meas = pct(raw[k, I["ids"]], c["pred"]["ID"])
            predd = 100.0 * (ratio ** (p_ids - 1.0) - 1.0)
            print(f"  {c['spec']:>3} {str(c['kind']):<9}{ratio:11.2f}"
                  f"{fmt(fing,'m'):>10}{meas:9.3f}%{predd:10.3f}%"
                  f"{meas-predd:10.3f}%")
        print()
        print("  'left over' is what width scaling does NOT explain.")

    # ---- the verdict -------------------------------------------------
    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if worst:
        worst.sort(reverse=True)
        e, s, key = worst[0]
        allv = np.array([w[0] for w in worst])
        print(f"  largest raw miss  : {e:.3f} % on {key} in spec {s}")
        print(f"  median raw miss   : {np.median(allv):.4f} % "
              f"over {allv.size} asked quantities")
        # The RAW miss is not the tool's error. A device can only have a
        # whole number of fingers, and the specification's width almost
        # never wants one, so the width-setting target carries the
        # difference -- up to half a finger in ng, which at 30 fingers is
        # 1.5 %. The line above used to stop there and announce 1.493 %
        # as 'the largest miss against a specification' while the detail
        # three screens up said 1.49 % of it was the rounding and 0.002 %
        # was left. That made a correct tool look two orders of magnitude
        # worse than it is.
        if worst_lhs:
            worst_lhs.sort(reverse=True)
            lv = np.array([w[0] for w in worst_lhs])
            e2, s2, k2, _ = worst_lhs[0]
            print()
            print(f"  AFTER the finger rounding, which is layout and not "
                  f"error:")
            print(f"  largest residual  : {e2:.3f} % on {k2} in spec {s2}")
            print(f"  median residual   : {np.median(lv):.4f} %")
            onn = [w for w in worst_lhs if w[3] is True]
            off = [w for w in worst_lhs if w[3] is False]
            if onn and off:
                a = max(w[0] for w in onn)
                b = max(w[0] for w in off)
                print(f"  split by where L landed:")
                print(f"    L on a stored node   ({len(onn):2d} targets) "
                      f": worst {a:.3f} %")
                print(f"    L inside a cell      ({len(off):2d} targets) "
                      f": worst {b:.3f} %")
                if b > 3 * a:
                    print("  the gap between those two is the L-axis "
                          "interpolation error and")
                    print("  nothing else: the solve, the width scaling "
                          "and the edge correction")
                    print("  cannot tell whether L fell on a node.")
    print(f"  geometry problems : {bad_geom}")
    if 'bad_p' in dir():
        print(f"  W-exponent disagreements : {bad_p}")
    print()
    print("  a miss here is the WHOLE chain: table, interpolation,")
    print("  the contour solve, the width scaling and the effective-width")
    print("  correction. Nothing in it is checked against itself.")

    # ---- leave the geometry laws behind, in the table -----------------
    # Only if the studies actually ran: a verify built from
    # specifications alone has nothing to record, and writing a file of
    # four identity fields would make gmid_size quote an empty measurement
    # as a measurement. Absent beats hollow.
    if "id_per_ng_spread_pct" in LAWS or "finger_width_effect_pct" in LAWS:
        LAWS["source"] = ("check_verify.py sections 4A/4B/4D, measured "
                          "against ngspice on this table's own device")
        with open(os.path.join(spec["lutdir"], "finger_laws.json"),
                  "w") as fh:
            json.dump(LAWS, fh, indent=1)
        print()
        print(f"  wrote {spec['lutdir']}/finger_laws.json -- gmid_size.py "
              f"reads it and")
        print("  quotes THIS table's geometry numbers in place of another "
              "device's.")


if __name__ == "__main__":
    main()
