#!/usr/bin/env python3
"""
=============================================================================
ADT-clone | MONTE CARLO AGAINST THE MISMATCH FORMULA, part 2 of 2
=============================================================================
    python3 check_mc.py

Reads mc_cases.json and mc_spice.txt. Five questions, in the order in
which a wrong answer would invalidate the ones after it:

  0. Did the samples differ at all, and are the devices independent?
     If reset did not redraw, every row is the same device and sigma is
     zero. If one draw is shared between instances, the three currents
     are correlated. Either makes everything below meaningless.
  1. The model's own perturbations: sigma(leff), sigma(weff), sigma(vth)
     against S_DL, S_DW and A_VT/sqrt(W.L).
  2. dlnID/dlnL, MEASURED by regressing ln(ID) on ln(leff) across the
     samples, against the value the table gave. This is the number the
     length term rests on, obtained without the table.
  3. sigma(ID)/ID against SIGID: the whole formula.
  4. Is the distribution gaussian? Where current depends exponentially
     on the threshold it need not be, and a sigma is then only half the
     story.

A Monte Carlo sigma from N samples is itself uncertain by a fraction
1/sqrt(2(N-1)), about 2.2 % at N = 1000. Every comparison is judged
against THAT, not against zero: a formula that agrees to 1 % with a
measurement good to 2 % has not been shown to be better than 2 %.
"""

import json
import sys

import numpy as np


def main():
    with open("mc_cases.json") as fh:
        spec = json.load(fh)
    devs, save = spec["devices"], spec["save"]
    nd, ns = len(devs), len(save)
    raw = np.loadtxt("mc_spice.txt")
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] == nd * ns + 1:           # wr_singlescale scale column
        raw = raw[:, 1:]
    if raw.shape[1] != nd * ns:
        sys.exit(f"mc_spice.txt has {raw.shape[1]} columns, expected "
                 f"{nd*ns} or {nd*ns+1}")
    N = raw.shape[0]
    if N < 30:
        sys.exit(f"only {N} samples: not enough to estimate a sigma")
    ok = np.all(np.isfinite(raw), axis=1)
    if not ok.all():
        print(f"  dropping {int((~ok).sum())} non-finite samples")
        raw = raw[ok]
        N = raw.shape[0]
    X = raw.reshape(N, nd, ns)
    col = {c: i for i, c in enumerate(save)}
    rel = 1.0 / np.sqrt(2.0 * (N - 1))        # uncertainty of a sigma

    print("=" * 76)
    print(f"  table {spec['lutdir']}   models {spec['corner']}")
    print(f"  {N} samples   a sigma from this many is good to "
          f"+-{100*rel:.1f} %")
    print("=" * 76)

    ids = np.abs(X[:, :, col["ids"]])

    # ---- 0 ----------------------------------------------------------------
    print("\n0. DID THE SAMPLES DIFFER, AND ARE THE DEVICES INDEPENDENT")
    dead = False
    for j, d in enumerate(devs):
        nu = len(np.unique(ids[:, j]))
        print(f"    {d['name']}: {nu} distinct currents in {N} samples")
        if nu < 0.9 * N:
            dead = True
    if dead:
        print("  SAMPLES REPEAT. reset is not redrawing the device, or the")
        print("  seed is not taking. Nothing below means anything.")
        sys.exit(1)
    lim = 3.0 / np.sqrt(N)
    worst = 0.0
    for a in range(nd):
        for b in range(a + 1, nd):
            r = np.corrcoef(np.log(ids[:, a]), np.log(ids[:, b]))[0, 1]
            worst = max(worst, abs(r))
            print(f"    corr(ID {devs[a]['name']}, ID {devs[b]['name']})"
                  f" = {r:+.3f}")
    print(f"    3-sigma limit for independent devices: {lim:.3f}   "
          + ("INDEPENDENT" if worst < lim else
             "<-- CORRELATED: one draw is being shared between instances"))

    # ---- 1 ----------------------------------------------------------------
    print("\n1. THE MODEL'S OWN PERTURBATIONS")
    print(f"    {'':3}{'quantity':<12}{'measured':>13}{'formula':>13}"
          f"{'ratio':>9}   within noise?")

    def line(name, q, meas, pred, unit, scale):
        if pred is None or not np.isfinite(pred) or pred == 0:
            print(f"    {name:3}{q:<12}{meas*scale:13.4g}{'--':>13}")
            return None
        r = meas / pred
        z = (r - 1.0) / rel
        flag = "yes" if abs(z) < 3 else f"NO ({z:+.1f} sigma)"
        print(f"    {name:3}{q:<12}{meas*scale:11.4g} {unit:<2}"
              f"{pred*scale:11.4g} {unit:<2}{r:8.3f}   {flag}")
        return z

    zs = []
    for j, d in enumerate(devs):
        p = d["pred"]
        leff, weff = X[:, j, col["leff"]], X[:, j, col["weff"]]
        vth = X[:, j, col["vth"]]
        zs.append(line(d["name"], "sigma(leff)", leff.std(ddof=1),
                       spec["S_DL"], "nm", 1e9))
        # weff is PER FINGER. The subcircuit randomises the TOTAL width
        # by S_DW and then hands the device w/ng, so the perturbation
        # arrives divided by the finger count. Comparing the per-finger
        # spread against the total-width coefficient reported a 25-fold
        # miss at ng = 24 and a 90-fold one at ng = 86 -- which is to
        # say, it measured ng. That is the law confirming itself, not a
        # discrepancy, and the ratio below is the one that means
        # something.
        zs.append(line(d["name"], "sigma(weff)/fg", weff.std(ddof=1),
                       spec["S_DW"] / d["ng"], "nm", 1e9))
        # vth also moves with leff, through the same short-channel
        # effect the length term is about, so the bare A_VT prediction
        # is only the delvto part. The regression removes the leff part
        # before comparing.
        # d(leff) = d(L drawn), but leff is SHORTER than L by a fixed
        # offset, so d ln(leff) is not d ln(L). Dividing the deviation by
        # the DRAWN length gives the same variable the table's
        # derivative is taken in; using ln(leff) would overstate every
        # slope by L/leff, which is 10 % at 1 um and more at 0.45.
        dl = (leff - leff.mean()) / d["L"]
        b = np.polyfit(dl, vth, 1)
        res = vth - np.polyval(b, dl)
        zs.append(line(d["name"], "sigma(vth)*", res.std(ddof=1),
                       p["SIGVT"], "mV", 1e3))
        print(f"       dVth/dlnL from the samples: {b[0]*1e3:+.1f} mV per "
              f"unit ln(L)   (raw sigma(vth) "
              f"{vth.std(ddof=1)*1e3:.3f} mV)")
    print("    * with the part explained by leff regressed out")
    print("    /fg per finger: the 3 nm lands on the TOTAL width, so the")
    print("      device sees S_DW/ng")

    # ---- 2 ----------------------------------------------------------------
    print("\n2. dlnID/dlnL, MEASURED FROM THE SAMPLES")
    print(f"    {'':3}{'from table':>12}{'from MC':>12}{'+-':>8}"
          f"{'textbook':>10}   verdict")
    for j, d in enumerate(devs):
        leff = X[:, j, col["leff"]]
        x = (leff - leff.mean()) / d["L"]      # d ln(L DRAWN), see above
        y = np.log(ids[:, j]) - np.log(ids[:, j]).mean()
        b = float((x * y).sum() / (x * x).sum())
        resid = y - b * x
        se = float(resid.std(ddof=2) / (x.std(ddof=1) * np.sqrt(N)))
        t = d["pred"]["_dlnI_dlnL"]
        z = (b - t) / se if se > 0 else float("nan")
        v = ("agrees with the table" if abs(z) < 3 else
             f"table is off by {z:+.1f} standard errors")
        # agreeing with the table means little if the run could not have
        # told the table from the textbook either
        zt = (b + 1.0) / se if se > 0 else float("nan")
        v += (", and rules out -1" if abs(zt) >= 3 else
              ", but cannot tell it from -1 at this N")
        d["_slope"], d["_slope_se"] = b, se
        print(f"    {d['name']:3}{t:12.3f}{b:12.3f}{se:8.3f}{-1.0:10.1f}"
              f"   {v}")
    print("    the regression sees only leff; every other source is noise")
    print("    to it, which is what the +- column is.")

    # ---- 3 ----------------------------------------------------------------
    print("\n3. sigma(ID)/ID AGAINST THE FORMULA")
    print(f"    {'':3}{'measured':>10}{'formula':>10}{'ratio':>8}"
          f"{'z':>7}   breakdown of the formula (% on ID)")
    bad = 0
    for j, d in enumerate(devs):
        p = d["pred"]
        meas = 100.0 * ids[:, j].std(ddof=1) / ids[:, j].mean()
        r = meas / p["SIGID"]
        z = (r - 1.0) / rel
        bad += int(abs(z) >= 3)
        vt = p["GM/ID"] * p["SIGVT"] * 100.0
        print(f"    {d['name']:3}{meas:9.3f}%{p['SIGID']:9.3f}%{r:8.3f}"
              f"{z:+7.1f}   Vt {vt:.3f}  beta {p['SIGBETA']:.3f}  "
              f"L {p['SIGL']:.3f}  W {p['SIGW']:.4f}")
    print(f"    |z| < 3 is agreement within the Monte Carlo's own noise.")

    # ---- 4 ----------------------------------------------------------------
    print("\n4. IS IT GAUSSIAN")
    print(f"    {'':3}{'skew of ID':>12}{'skew of ln ID':>15}"
          f"{'3-sigma limit':>15}")
    slim = 3.0 * np.sqrt(6.0 / N)
    for j, d in enumerate(devs):
        def skew(v):
            c = v - v.mean()
            return float((c ** 3).mean() / (c ** 2).mean() ** 1.5)
        a, b = skew(ids[:, j]), skew(np.log(ids[:, j]))
        note = ""
        if abs(a) > slim and abs(b) < abs(a):
            note = "   ID is closer to LOG-normal here"
        print(f"    {d['name']:3}{a:12.3f}{b:15.3f}{slim:15.3f}{note}")

    # ---- nominal ----------------------------------------------------------
    try:
        nom = np.loadtxt("mc_nominal.txt")
        nom = np.atleast_2d(nom)
        if nom.shape[1] == nd * ns + 1:
            nom = nom[:, 1:]
        nom = nom.reshape(nd, ns)
        print("\n   NOMINAL DEVICE (mm_ok = 0) AGAINST THE TABLE AND THE MEAN")
        for j, d in enumerate(devs):
            i0 = abs(nom[j, col["ids"]])
            print(f"    {d['name']}: nominal {i0*1e6:.4f} uA   table "
                  f"{d['pred']['ID']*1e6:.4f} uA "
                  f"({100*(i0/d['pred']['ID']-1):+.3f} %)   MC mean "
                  f"{ids[:, j].mean()*1e6:.4f} uA "
                  f"({100*(ids[:, j].mean()/i0-1):+.3f} %)")
    except Exception as e:                                   # noqa: BLE001
        print(f"\n   no usable mc_nominal.txt ({type(e).__name__}); the "
              f"mm_ok = 0 branch did not produce a row.")

    print("\n" + "=" * 76)
    if bad == 0:
        print("  VERDICT: the formula agrees with the model at all "
              f"{nd} devices, to within")
        print(f"  the +-{100*rel:.1f} % a {N}-sample sigma can resolve.")
    else:
        print(f"  VERDICT: {bad} of {nd} devices disagree with the formula "
              "by more than the")
        print("  Monte Carlo's own noise. Sections 1 and 2 say which term.")
    print("=" * 76)


if __name__ == "__main__":
    main()
