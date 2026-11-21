#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | MISMATCH COEFFICIENTS, READ FROM THE PDK
=============================================================================
    python3 make_mismatch.py lut_nmos_hv_tt_27
    python3 make_mismatch.py lut_nmos_hv_tt_27 --models /path/to/models

Writes mismatch.json beside the table. No simulation.

WHY THIS IS A SCRIPT AND NOT FOUR CONSTANTS IN A FILE

Four numbers would fit in a line of Python, and that line would be wrong
the first time the PDK is updated, silently, because nothing would be
checking. It has already happened once in this project with the W-scaling
exponents: a corrected value in one place and the old guess in another.

So this reads the numbers from the PDK, and -- this is the part that
matters -- it ALSO reads the formula that applies them and refuses to
write anything if that formula is not the one the sizing maths assumes.
Those assumptions are:

  * sigma(delta Vt) = delvto_mm / sqrt(m * l * w * 1e12), with l and w in
    metres, so the 1e12 makes the area um^2 and delvto_mm is in V.um.
  * w passed to the subcircuit is the TOTAL width, not the width of one
    finger. The subcircuit divides it by ng itself to get the finger, so
    the mismatch area is W_total x L and the FINGER COUNT DOES NOT ENTER.
  * the width and length perturbations are ABSOLUTE -- no division by the
    area -- so they are an edge-roughness floor that no amount of width
    buys down.
  * the third argument of agauss is the number of sigmas the second one
    represents, and it is 1, so these are one-sigma values.

Every one of those is checked against the text of the model file. If the
PDK changes any of them, this stops with the line it did not recognise
rather than handing the sizer a number whose meaning has moved.

WHAT IT DOES NOT DO: it does not evaluate the formula, size anything, or
run SPICE. It extracts constants and proves they mean what we think.
"""

import argparse
import json
import os
import re
import sys

DEFAULT_MODELS = "/foss/pdks/ihp-sg13g2/libs.tech/ngspice/models"

# the four coefficients, and what each one is
WANTED = {
    "delvto_mm": "threshold voltage, V.um, 1 sigma, divided by sqrt(area)",
    "factuo_mm": "mobility factor, fraction.um, 1 sigma, divided by "
                 "sqrt(area)",
    "dw_mm": "width, metres, 1 sigma, ABSOLUTE (no area scaling)",
    "dl_mm": "length, metres, 1 sigma, ABSOLUTE (no area scaling)",
}

# The expected shape of the two area-scaled expressions. Whitespace is
# stripped before matching; everything else must be exact.
AREA_DEN = "sqrt(m*l*w*1e12)"
SIG_ARG = "(mm_ok!=1?0:1)"


def squash(s):
    return re.sub(r"\s+", "", s)


def read(path):
    with open(path, errors="replace") as fh:
        return fh.read()


def find_files(models, tail):
    """the two files that mention this device's mismatch parameters"""
    par = mod = None
    for fn in sorted(os.listdir(models)):
        if not fn.endswith(".lib"):
            continue
        txt = read(os.path.join(models, fn))
        if f"{tail}_delvto_mm" not in txt:
            continue
        if ".param" in txt and "agauss" not in txt:
            par = par or os.path.join(models, fn)
        if "agauss" in txt:
            mod = mod or os.path.join(models, fn)
    return par, mod


def params_of(txt, tail):
    """the four .param lines, as floats"""
    out, where = {}, {}
    for key in WANTED:
        m = re.search(rf"^\s*\.param\s+(\S*{re.escape(tail)}_"
                      rf"{re.escape(key)})\s*=\s*([-\d.eE+]+)",
                      txt, re.MULTILINE)
        if not m:
            raise SystemExit(
                f"  no .param for {tail}_{key} in the parameter file.\n"
                f"  Not guessing a value for it.")
        out[key] = float(m.group(2))
        where[key] = m.group(1)
    return out, where


def check_law(txt, tail, names):
    """
    Prove the formula is the one the sizing maths assumes.

    Returns a list of (claim, ok, detail). Nothing is written unless every
    claim holds, because a coefficient whose formula has moved is worse
    than no coefficient: it is a wrong number that looks right.
    """
    s = squash(txt)
    out = []

    # The parameter's FULL name, as the parameter file declares it --
    # not rebuilt from the flavour. The first version of this rebuilt it
    # and failed every check against a file that was perfectly correct,
    # which is the checker working: it refused to write rather than
    # assume.
    for key in ("delvto", "factuo"):
        nom = "0" if key == "delvto" else "1"
        want = (f"{key}='agauss({nom},{names[key + '_mm']}/{AREA_DEN},"
                f"{SIG_ARG})'")
        out.append((f"{key} scaled by 1/sqrt(m*l*w*1e12), 1 sigma",
                    want in s, want))

    for key in ("w", "l"):
        want = f"{key}='agauss({key},{names['d' + key + '_mm']},{SIG_ARG})'"
        out.append((f"d{key} is ABSOLUTE, not divided by the area",
                    want in s, want))

    # w is the TOTAL width: the subcircuit divides it by ng itself. If it
    # were already the finger width this would say max(w,wmin).
    out.append(("w passed in is the TOTAL width (subckt uses w/ng)",
                "max(w/ng,wmin)" in s, "max(w/ng,wmin)"))

    # the finger count reaches the device as nf, and is absent from the
    # mismatch denominator, which is the whole reason area and finger
    # count decouple
    out.append(("ng reaches the device as nf, and m as mult",
                "nf='ng'mult='m'" in s, "nf='ng' mult='m'"))
    out.append(("ng does NOT appear in the mismatch denominator",
                "ng" not in AREA_DEN, AREA_DEN))

    # the parity branch: this is the same law the junction tables
    # measured, written in the PDK's own hand
    out.append(("the subcircuit branches on the PARITY of ng",
                "floor(floor(ng/2+0.501)*2+0.001)!=ng" in s,
                "floor(floor(ng/2+0.501)*2+0.001) != ng"))
    return out


def defaults_of(txt, dev):
    """z1, z2 and wmin from the .subckt line, for the record"""
    m = re.search(rf"\.subckt\s+{re.escape(dev)}\b(.*?)(?=\n\s*\n|\n\.)",
                  txt, re.DOTALL)
    blob = squash(m.group(1)) if m else ""
    got = {}
    for k in ("z1", "z2", "wmin"):
        mm = re.search(rf"{k}=([-\d.eE+]+)", blob)
        if mm:
            got[k] = float(mm.group(1))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lutdir")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--device", default=None,
                    help="subcircuit name; default is the table's own")
    a = ap.parse_args()

    with open(os.path.join(a.lutdir, "meta.json")) as fh:
        meta = json.load(fh)
    dev = a.device or meta["device"]

    # sg13_hv_nmos -> hv_nmos, which is what the parameter names carry
    m = re.search(r"((?:hv|lv)_[np]mos)", dev)
    if not m:
        sys.exit(f"  cannot tell the flavour from device name {dev!r}.\n"
                 f"  Pass --device, e.g. --device sg13_hv_nmos")
    tail = m.group(1)

    if not os.path.isdir(a.models):
        sys.exit(f"  no such directory: {a.models}\n"
                 f"  Pass --models with the PDK's ngspice models folder.")

    print("=" * 72)
    print(f"  MISMATCH   {a.lutdir}")
    print(f"  device  : {dev}   (parameters named *_{tail}_*)")
    print(f"  models  : {a.models}")
    print("=" * 72)

    par, mod = find_files(a.models, tail)
    if not par or not mod:
        sys.exit(f"  could not find both the parameter file and the model "
                 f"file\n  mentioning {tail}_delvto_mm in {a.models}")
    print(f"  coefficients from : {os.path.basename(par)}")
    print(f"  formula from      : {os.path.basename(mod)}")

    ptxt, mtxt = read(par), read(mod)
    vals, names = params_of(ptxt, tail)

    print("\n  COEFFICIENTS")
    for k, v in vals.items():
        print(f"    {names[k]:36} = {v:<10g}  {WANTED[k]}")

    print("\n  THE FORMULA, CHECKED AGAINST THE MODEL FILE")
    claims = check_law(mtxt, tail, names)
    bad = 0
    for claim, ok, detail in claims:
        print(f"    [{'ok' if ok else 'NO'}] {claim}")
        if not ok:
            bad += 1
            print(f"         expected to find: {detail}")
    if bad:
        sys.exit(f"\n  {bad} assumption(s) NOT confirmed. Writing nothing.\n"
                 f"  The sizing maths is built on these; a coefficient "
                 f"whose formula\n  has moved is worse than no coefficient, "
                 f"because it looks right.")

    geo = defaults_of(mtxt, dev)
    if geo:
        print("\n  SUBCIRCUIT DEFAULTS (the junction geometry, for the "
              "record)")
        for k, v in geo.items():
            print(f"    {k:5} = {v:g}")

    out = dict(
        device=dev, flavour=tail,
        A_VT=vals["delvto_mm"],          # V.um
        A_BETA=vals["factuo_mm"],        # fraction.um
        S_DW=vals["dw_mm"],              # m, absolute
        S_DL=vals["dl_mm"],              # m, absolute
        sigmas=1,
        area="W_drawn * L_drawn, in um^2; the FINGER COUNT does not enter",
        law="sigma(dVt) = A_VT / sqrt(W_um * L_um); "
            "sigma(dbeta/beta) = A_BETA / sqrt(W_um * L_um); "
            "sigma(dW) = S_DW and sigma(dL) = S_DL, absolute",
        geometry=geo,
        source=dict(params=par, model=mod, names=names),
    )
    path = os.path.join(a.lutdir, "mismatch.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    av = vals["delvto_mm"] * 1e3
    print(f"\n  wrote {path}")
    print(f"\n  A_VT = {av:g} mV.um, so a device of W x L = 1 um^2 has")
    print(f"  sigma(dVt) = {av:g} mV, and every doubling of AREA divides")
    print(f"  that by sqrt(2). sigma(dL) = {vals['dl_mm']*1e9:g} nm does "
          f"NOT scale:")
    # The shortest length this table actually holds, not a literal. The
    # literal here used to be 0.45 um, which is the hv device's minimum,
    # and on the lv table it understated the floor by a factor of 3.5 --
    # quoting 0.444 % where the device's own shortest gate gives 1.54 %.
    # The floor is the one mismatch number no width can buy down, so
    # understating it is the worst direction to be wrong in.
    lmin = float(min(meta["L"]))
    print(f"  at L = {lmin*1e6:g} um, the shortest this table holds, it is "
          f"{vals['dl_mm']/lmin*100:.3f} % on")
    print(f"  the current whatever the width.")


if __name__ == "__main__":
    main()
