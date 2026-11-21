#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | LUT ENGINE
Reads a directory produced by pack_lut.py and answers queries.
=============================================================================

    from gmid_lut import LUT
    lut = LUT("lut_nmos_hv_tt_27")
    lut.eval("GM/ID", L=0.83e-6, VGS=1.137, VDS=0.9, VSB=0.25)

Four things make this more than plain multilinear interpolation, and all
four were needed to keep the weak-inversion region honest:

  * every axis is interpolated with a CUBIC Hermite whose node slopes
    come from the parabola through three neighbouring nodes, so it is
    valid on the non-uniform L axis and one-sided, still second order,
    at the ends. Measured on the real table this cut the gds error 5.8x
    without one extra simulation;

  * each axis is combined in the domain where the physics is nearly
    straight: VGS and VSB in the LOG domain, where weak-inversion
    current is a true exponential, and VDS in the LINEAR domain, where
    near VDS = 0 the current is a power law and the log domain is 11x
    worse. On a textbook triode law this is 825x on deep-triode p90;


  * the L axis is interpolated GEOMETRICALLY, because essentially every
    MOSFET parameter goes like 1/L: a straight line in log-log, a badly
    curved one in linear-linear;

  * strictly positive parameters are interpolated in the LOG domain. In
    weak inversion ID, gm and gds are exponential in VGS and in VSB
    (through VTH), and log interpolation is exact for a pure exponential;

  * a query whose 16 surrounding grid points are not all present returns
    NaN. The grid is incomplete by construction (the SOA clip), and
    silently substituting a clipped neighbour would quietly corrupt every
    chart built on top of it.

Width scaling is done on EFFECTIVE width, not drawn width:

    Weff(w, ng) = w - ng * dW        dW = 0.06 um per finger, measured

so asking for a device twice as strong really gives one twice as strong.
"""

import json
import os
import re

import numpy as np

# -----------------------------------------------------------------------
# derived quantities.  'p' is the power of (W/Wref) the quantity carries.
# -----------------------------------------------------------------------
DERIVED = {
    "ID":      (lambda d: d["ids"],                                     1),
    "IG":      (lambda d: d["igs"] + d["igd"] + d["igb"],               1),
    "GM":      (lambda d: d["gm"],                                      1),
    "GMB":     (lambda d: d["gmb"],                                     1),
    "GDS":     (lambda d: d["gds"],                                     1),
    "VTH":     (lambda d: d["vth"],                                     0),
    "VOV":     (lambda d: d["vgt"],                                     0),
    "VDSAT":   (lambda d: d["vdss"],                                    0),
    "RO":      (lambda d: 1.0 / d["gds"],                              -1),
    "VA":      (lambda d: d["ids"] / d["gds"],                          0),
    "LAMBDA":  (lambda d: d["gds"] / d["ids"],                          0),
    "GM/ID":   (lambda d: d["gm"] / d["ids"],                           0),
    "VSTAR":   (lambda d: 2.0 * d["ids"] / d["gm"],                     0),
    "GM/GDS":  (lambda d: d["gm"] / d["gds"],                           0),
    "GMB/GM":  (lambda d: d["gmb"] / d["gm"],                           0),
    # fug is the model's own unity-gain frequency. Verified on this PDK to
    # be exactly gm / (2*pi*(cgg+cgsol+cgdol)) -- i.e. it INCLUDES the
    # overlap capacitance, which is 58 % on top of cgg at W_ref. Computing
    # fT from cgg alone overestimates it by that much.
    "FT":      (lambda d: d["fug"],                                     0),
    "CGG":     (lambda d: d["cgg"],                                     1),
    "CGS":     (lambda d: d["cgs"],                                     1),
    "CGD":     (lambda d: d["cgd"],                                     1),
    "CGB":     (lambda d: d["cgb"],                                     1),
    "CDG":     (lambda d: d["cdg"],                                     1),
    "CDB":     (lambda d: d["cdb"],                                     1),
    "CSG":     (lambda d: d["csg"],                                     1),
    "CSB":     (lambda d: d["csb"],                                     1),
    "CGSOL":   (lambda d: d["cgsol"],                                   1),
    "CGDOL":   (lambda d: d["cgdol"],                                   1),
    "CJD":     (lambda d: d["cjd"],                                     1),
    "CJS":     (lambda d: d["cjs"],                                     1),
    # total gate capacitance including overlaps: the one that sets fT
    "CGG_TOT": (lambda d: d["cgg"] + d["cgsol"] + d["cgdol"],           1),
    "CDTOT":   (lambda d: d["cdb"] + d["cjd"] + d["cgdol"],             1),
    "VEARLY":  (lambda d: d["vearly"],                                  0),
    "BEFF":    (lambda d: d["beff"],                                    1),
    # -1 is MEASURED, not a dimensional guess: study A varied the finger
    # COUNT at a fixed finger WIDTH -- the one experiment RAW_P governs --
    # and found 13.03 ohm at ng = 1 and 260.5 mohm at ng = 50, so
    # p = -1.0000 with residual 8.9e-16, identically on sg13_hv_nmos and
    # sg13_lv_nmos. The full argument, including why the older p = -0.035
    # measured a DIFFERENT experiment, is at RAW_P['rg'] below.
    #
    # This comment used to assert the opposite while the value read -1,
    # because the value was corrected from measurement and the prose was
    # not. DERIVED is checked before RAW_P in eval(), so the two must
    # agree; the clash check below enforces the VALUES and nothing
    # enforces the prose, which is why this paragraph is explicit.
    "RG":      (lambda d: d["rg"],                                     -1),
    "SFL":     (lambda d: d["sfl"],                                     1),
    "SID":     (lambda d: d["sid"],                                     1),
    "FKNEE":   (lambda d: d["fknee"],                                   0),
    # ---- noise, input-referred -----------------------------------------
    # sid is the THERMAL part only (white); sfl is the flicker coefficient
    # such that S_flicker(f) = sfl/f. fknee = sfl/sid.
    # d["kt4"] is 4.k_B.T at THE TABLE'S OWN temperature, supplied by
    # eval() from meta.json. This used to be the literal 4 * 4.14195e-21,
    # k_B.T at 300.15 K -- correct while every table was built at 27 C,
    # and silently wrong from the day the -40 and 125 C tables existed:
    # gamma read 22 % low on a cold table and 33 % high on a hot one,
    # because sid scales with the real T and the divisor did not.
    "GAMMA":   (lambda d: d["sid"] / (d["kt4"] * d["gm"]),              0),
    "SVG":     (lambda d: d["sid"] / d["gm"] ** 2,                      -1),
    "SVG_1HZ": (lambda d: (d["sid"] + d["sfl"]) / d["gm"] ** 2,        -1),
    "ID/W":    (lambda d: d["ids"],                                     0),   # special
    "GM/W":    (lambda d: d["gm"],                                      0),   # special
}

# ---------------------------------------------------------------------------
CORNER_OF = {"tt": "mos_tt", "ss": "mos_ss", "ff": "mos_ff",
             "sf": "mos_sf", "fs": "mos_fs"}


def corner_of_table(meta, lutdir="this table"):
    """
    Which .lib corner section a packed table was built from, taken from
    its own provenance: pack_lut.py records the raw file it packed in
    meta.json['source'].

    The tag is matched as a TOKEN anywhere in the file name, not as a
    suffix. The first version here required the name to end in '_tt',
    which worked for lut_ss.raw and refused lut_nmos_hv_tt_27.raw, where
    the tag sits in the middle. Two tags, or none, is a refusal: probing
    or verifying a table against the wrong models produces a full page of
    believable numbers that mean nothing, so guessing is worse than
    stopping.

    Lives here, in one place, because make_probe.py and make_verify.py
    both need it and two copies of one rule is how the rule drifts.
    """
    import re
    src = str(meta.get("source", ""))
    stem = os.path.splitext(os.path.basename(src))[0]
    toks = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", stem) if t]
    found = sorted({CORNER_OF[t] for t in toks if t in CORNER_OF})
    if len(found) == 1:
        return found[0], src
    if len(found) > 1:
        raise ValueError(
            f"the name {src!r} of the file {lutdir} was packed from "
            f"contains more than one corner tag ({', '.join(found)}), "
            f"so which models to use is ambiguous. Rename it.")
    raise ValueError(
        f"cannot tell which corner {lutdir} was built from.\n"
        f"  meta.json source = {src!r}\n"
        f"  the name needs one of {', '.join(sorted(CORNER_OF))} as a "
        f"word in it, e.g. lut_nmos_hv_tt_27.raw or lut_ss.raw\n"
        f"  refusing to guess: the wrong models give numbers that look "
        f"fine and are not."
    )

# ---------------------------------------------------------------------------
# W-power of each RAW (lower-case) parameter name.
#
# eval() accepts raw names as well as the DERIVED keys, and it used to
# multiply EVERY raw parameter by the width ratio -- i.e. assume p = 1 for
# all of them. That is right for a current or a capacitance and wrong for
# vth, vdss and fug, which do not depend on W at all: asking for
# eval('vth', W=...) came back scaled by W/W_ref. Caught by a round trip
# in which a perfect model disagreed with itself by 103 %, 309 % and
# 1134 % on exactly those three.
#
# Nothing computed before this was affected: check_probe, gmid_size and the
# verifier all go through the upper-case DERIVED keys, which carry their
# own correct powers.
#
# Most entries are taken straight from DERIVED, which is authoritative.
# The four marked INFERRED are dimensional reasoning, not measurement;
# make_verify.py measures all of them against the model at W and 2W, so
# they stop being assumptions as soon as that runs.
# ---------------------------------------------------------------------------
RAW_P = {
    # extensive: proportional to W
    "ids": 1, "gm": 1, "gmb": 1, "gds": 1,
    "cgg": 1, "cgs": 1, "cgd": 1, "cgb": 1, "cdg": 1, "cdb": 1,
    "csg": 1, "csb": 1, "cgsol": 1, "cgdol": 1, "cjd": 1, "cjs": 1,
    "beff": 1, "sfl": 1, "sid": 1,
    "igs": 1, "igd": 1, "igb": 1,
    "sig": 1,                     # MEASURED p = 1.0000, residual 1e-9
    "iavl": 1,                    # MEASURED p = 1.0000, residual 3e-9
    # AFFINE, not a power law: ng fingers share their source/drain
    # diffusions, so there are ng+1 junctions and not 2*ng, and these
    # carry a fixed end term as well as a per-finger one. Measured on
    # Cgg_tot:  C(ng) = 6.8973*ng + 0.4869 fF fits to 0.010 %, while
    # pure proportionality is out by 5.3 % at ng = 1. Kept at 1 because
    # the per-finger term dominates and the aggregate effect on Cgg_tot
    # and fT is about 1.4 %; the fix is one extra sweep at ng = 1, which
    # would give the end term over the whole grid.
    "cgsol": 1,                   # MEASURED 0.917, residual 8e-2: affine
    "cgdol": 1,                   # MEASURED 0.914, residual 8e-2: affine
    "cjd": 1,                     # MEASURED 0.888, residual 3e-1: affine
    "cjs": 1,                     # MEASURED 0.829, residual 2e-1: affine
    "ijd": 1,                     # MEASURED 0.963, residual 9e-2: affine
    # weff as STORED is the width of ONE FINGER, so it does not change
    # with the finger count at all. It was set to 1 here while thinking
    # of the total width. Measured p = 0.0000, residual 2e-15.
    "weff": 0,
    # intensive: independent of W
    "vth": 0, "vgt": 0, "vdss": 0, "vearly": 0, "fknee": 0,
    # fug = gm / (2.pi.(cgg + cgsol + cgdol)): the numerator is exactly
    # proportional to ng and the denominator is affine, so fug drifts
    # about 1.4 % over the useful ng range. MEASURED p = 0.016.
    "fug": 0,
    "cigid": 0,                   # INFERRED: a correlation coefficient
    "vgs": 0, "vds": 0, "vsb": 0, "leff": 0,
    # rg: TWO measurements existed and they contradicted each other.
    #
    #   the old one  40.7 ohm at ng = 1, 35.11 ohm at ng = 50,
    #                p = -0.035, residual 3.5e-2  -> nearly flat, and
    #                not a power law at all
    #   study A      13.03 ohm at ng = 1, 260.5 mohm at ng = 50,
    #                p = -1.0000, residual 8.9e-16 -> exactly 1/ng,
    #                and a perfect power law
    #
    # Study A wins, for a reason and not by being newer. RAW_P is
    # applied in exactly one situation: rescaling a stored quantity from
    # the reference FINGER COUNT to another at the SAME FINGER WIDTH.
    # That is what study A varies and what the old measurement did not;
    # at a fixed TOTAL width every finger narrows as the count grows,
    # which is a different experiment with a different answer. It is
    # also measured identically on sg13_hv_nmos and sg13_lv_nmos, which
    # a coincidence would not be.
    #
    # The old comment warned that -1 'would have been wrong by a factor
    # of seven, silently'. The arithmetic was right and the sign of the
    # conclusion was not: with p = 0 a 5-finger reference scaled to 34
    # fingers does not move at all, while the device's rg falls by 34/5.
    # The factor of seven was what the OLD setting cost.
    "rg": -1,
}

# The same exponent lives in DERIVED (upper case) and RAW_P (lower
# case), and they drifted apart once already: RAW_P["rg"] was corrected
# from measurement while DERIVED["RG"] kept the old guess, and since
# eval() consults DERIVED first the correction did nothing. Checked here
# at import so the next divergence cannot be silent.
_clash = {k: (DERIVED[k][1], RAW_P[k.lower()])
          for k in DERIVED
          if k.lower() in RAW_P and DERIVED[k][1] != RAW_P[k.lower()]}
if _clash:
    raise AssertionError(
        "DERIVED and RAW_P disagree on the W-power of "
        + ", ".join(f"{k}: {a} vs {b}" for k, (a, b) in _clash.items()))
del _clash

# Parameters that are AFFINE in the finger count, not proportional:
# adjacent fingers share their source and drain diffusions, so ng
# fingers have ng+1 junctions and not 2*ng, and each of these carries a
# fixed end term as well as a per-finger one. Measured exponents
# 0.83 to 0.96 with log-log residuals of 1e-1 to 3e-1, i.e. not power
# laws at all. Correcting them needs a SECOND table at a different
# finger count; attach_edge() supplies it.
AFFINE = {"cgsol", "cgdol"}

# ---------------------------------------------------------------------------
# AND THE JUNCTIONS ARE NOT AFFINE EITHER. THEY FOLLOW THE PARITY.
#
# A device of ng fingers has ng+1 diffusion regions, and they alternate
# source, drain, source, drain ... Region 1 is always a source; region
# ng+1 is a source when ng is EVEN and a drain when ng is ODD. An inner
# region sits between two fingers and is shared by them; the two end
# regions serve one finger each and so are wider.
#
# That makes the junction count a function of the PARITY of ng, not an
# affine function of it:
#
#   drain,  ng even : ng/2 shared,        no end
#   drain,  ng odd  : (ng-1)/2 shared,    one end
#   source, ng even : ng/2 - 1 shared,    two ends
#   source, ng odd  : (ng-1)/2 shared,    one end
#
# This was found, not assumed. The affine law was fitted from the two
# tables at ng = 5 and ng = 1 -- BOTH ODD -- so it reproduced the odd
# sequence exactly and was simply wrong at every even count, which is
# why Cdtot read 0.000 % at ng = 11, 27, 35, 77 and 83 and -2.925 % at
# ng = 6, -0.658 % at 30, -0.366 % at 60. Measured against the model at
# ng = 1, 2, 5, 10, 25, 50:
#
#        cjd      affine +43.8 % at ng=2   ->  parity 0.03 %
#        cjs      affine -23.2 % at ng=2   ->  parity 0.03 %
#        ijd      affine +11.5 % at ng=2   ->  parity 0.02 %
#
# and it costs NO new simulation: the two existing tables give the two
# unknowns, because ng = 1 is 0 shared + 1 end and ng = 5 is 2 shared +
# 1 end, which is a non-singular pair.
#
# cdb is NOT in here: it is the intrinsic channel-to-bulk capacitance,
# not a junction, and it measures exactly proportional (p = 1.0000,
# residual 2e-9). The gate overlaps are not in here either: they ARE
# exactly affine at both parities (0.04 % over the same six counts),
# because every finger has one gate edge per side whatever the sharing.
# ---------------------------------------------------------------------------
JUNCTION = {"cjd": "D", "ijd": "D", "cjs": "S"}

# every parameter whose value depends on the finger count by some law
# other than plain proportionality
NG_LAW = AFFINE | set(JUNCTION)


def finger_regions(ng, term, parity=True):
    """
    (shared, end) diffusion-region counts of one terminal, for a device
    of ng fingers. term is 'D' or 'S'.

    With parity=False it returns the ODD branch for any ng, which is
    continuous in ng and exact at every odd count. That form exists for
    one reason: the sizing assistant iterates towards a finger count,
    and a law that steps between neighbouring integers turns that fixed
    point into a discontinuity and makes the iteration oscillate between
    an even and an odd answer forever. The device that actually gets
    built is always evaluated with parity=True.
    """
    ng = np.asarray(ng, float)
    if not parity:
        return (ng - 1.0) / 2.0, np.ones_like(ng)
    even = np.mod(np.round(ng), 2) == 0
    if term == "D":
        return (np.where(even, ng / 2.0, (ng - 1.0) / 2.0),
                np.where(even, 0.0, 1.0))
    return (np.where(even, ng / 2.0 - 1.0, (ng - 1.0) / 2.0),
            np.where(even, 2.0, 1.0))


ENV = {"np": np, "pi": np.pi, "sqrt": np.sqrt, "abs": np.abs,
       "log10": np.log10, "exp": np.exp}


def tokens(s):
    """
    The names in a chart expression, longest first.

    One implementation, used both by eval() to substitute values and by
    deps_of_expr() to work out what the expression depends on. Two
    tokenisers would drift, and the one that drifted would be the one
    deciding whether a warning is printed -- a silent disagreement
    between what a chart computes and what it claims to depend on.
    """
    slashed = sorted((k for k in set(DERIVED) | set(PER_W) if "/" in k),
                     key=len, reverse=True)
    pat = "|".join([re.escape(k) for k in slashed]
                   + [r"[A-Za-z_][A-Za-z_0-9]*"])
    return sorted(set(re.findall(pat, s, flags=re.IGNORECASE)),
                  key=len, reverse=True)


NUM = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")


def deps_of_expr(s, params=()):
    """
    Which stored parameters a whole EXPRESSION is built from.

    deps_of() answers for one key. A chart's y axis is usually not one
    key -- 'GM/ID*FT', 'SID/(4*k*T*GM)' -- and asking deps_of() about
    those gets None, 'assume the worst', which made every expression
    chart carry a finger-count warning whether or not it had any
    finger-count dependence. A warning that fires on everything is a
    warning nobody reads.

    Returns None, still meaning assume the worst, if ANY name in the
    expression cannot be accounted for. Being unable to analyse one
    token is not permission to ignore it.

    The exponent of a float literal is a name as far as the tokeniser is
    concerned -- 1.380649e-23 contributes 'e' -- so literals come out
    first. eval() does not need that, because an unresolvable token is
    simply left in place there; here it would be read as an unknown
    device quantity and poison the answer.
    """
    out = set()
    for t in tokens(NUM.sub(" ", str(s))):
        k = t.upper()
        if k in DERIVED or k in PER_W:
            d = deps_of(k)
            if d is None:
                return None
            out |= d
        elif t.lower() in params or t.lower() in RAW_P:
            out.add(t.lower())
        elif k in ("L", "W", "WEFF", "AREA", "VGS", "VDS", "VSB"):
            continue                      # an axis, not a stored param
        elif t in ENV:
            continue                      # a function or a constant
        else:
            return None
    return out


def deps_of(key):
    """
    Which STORED parameters a quantity is built from.

    Read out of the DERIVED lambda itself rather than kept in a second
    list beside it: the lambda is handed a dict that records every key it
    asks for and answers 1.0 to all of them, so the arithmetic runs and
    the dependency set falls out. This file has already been bitten twice
    by the same exponent living in two places (RAW_P['rg'] corrected
    while DERIVED['RG'] kept the old guess), and a hand-written
    dependency list is the same mistake waiting to happen -- a quantity
    added to DERIVED next month must be classified without anyone
    remembering to.

    Returns None for a key this cannot analyse. Callers must read that as
    'assume the worst', never as 'no dependencies'.
    """
    k = str(key).upper()
    if k in ("FT", "FUG"):
        # eval() does NOT read the stored fug once an edge table is
        # attached: it rebuilds fT from the corrected capacitances,
        # because a ratio of a proportional numerator to an affine
        # denominator is neither proportional nor affine. So fT's real
        # dependencies are those, and not fug.
        return {"gm", "cgg", "cgsol", "cgdol"}
    if k in DERIVED:
        seen = set()

        class _Rec(dict):
            def __missing__(s, kk):
                seen.add(kk)
                s[kk] = 1.0
                return 1.0

        DERIVED[k][0](_Rec())
        return seen
    if str(key).lower() in RAW_P:
        return {str(key).lower()}
    return None


PER_W = {"ID/W", "GM/W"}
# quantities that must stay linear: they can be negative or cross zero
NO_LOG = {"cgd", "cdb", "cdg", "csb", "csg", "cgb", "vgt", "ijd",
          "igs", "igd", "igb"}


class LUT:
    def __init__(self, path):
        with open(os.path.join(path, "meta.json")) as fh:
            m = json.load(fh)
        self.path = path
        self.meta = m
        self.device = m["device"]
        self.L = np.asarray(m["L"], float)
        self.VGS = np.asarray(m["VGS"], float)
        self.VDS = np.asarray(m["VDS"], float)
        self.VSB = np.asarray(m["VSB"], float)
        self.OS = np.asarray(m["OS"], np.int64)
        self.n_b = np.asarray(m["n_b"], np.int64)
        self.params = m["params"]
        self.W_ref_eff = float(m["W_ref"])      # 9.7 um effective
        self.W_nom = float(m["W_nom"])          # 10 um drawn
        self.ng = int(m["ng"])
        self.dW = (self.W_nom - self.W_ref_eff) / self.ng   # 0.06 um/finger
        # the L axis is interpolated in log space: nearly every parameter
        # goes like 1/L, straight in log-log and badly curved otherwise
        self._logL = np.log(self.L)
        self._cache = {}
        self.edge = None            # the ng = 1 table, if attached
        # Kelvin, from the table. An older table has no key and was
        # built at 27 C.
        self.temp_c = float(m.get("temp", 27.0))
        self.temp_k = self.temp_c + 273.15

    def attach_edge(self, path):
        """
        Attach the companion table taken at a DIFFERENT finger count, so
        the affine parameters can be split into their per-finger and
        fixed end terms:

            P(ng) = c*ng + e      c = (P5 - P1)/4   e = (5*P1 - P5)/4

        for a reference of 5 fingers and an edge table of 1. The edge
        table may be on a coarser voltage grid -- e is a few per cent
        correction, so 1 % on e is 0.07 % on the result -- and it is
        interpolated on its own grid by the same engine.
        """
        e = LUT(path)
        if e.ng >= self.ng:
            raise ValueError(
                f"the edge table has ng = {e.ng} and the main table "
                f"ng = {self.ng}; they must differ, and the edge one is "
                f"meant to be the smaller.")
        wf_m = self.W_ref_eff / self.ng
        wf_e = e.W_ref_eff / e.ng
        if abs(wf_m - wf_e) > 1e-12:
            raise ValueError(
                f"the two tables were built at different FINGER widths, "
                f"{wf_e*1e6:g}u and {wf_m*1e6:g}u. The whole point is that "
                f"only the COUNT differs, so this pair cannot be used.")
        if list(e.L) != list(self.L) or list(e.VSB) != list(self.VSB):
            raise ValueError("the two tables must share the L and VSB "
                             "axes; only the VGS/VDS grid may be coarser.")
        self.edge = e
        return self

    def affine_sensitive(self, key):
        """
        Does this quantity change with the FINGER COUNT at a fixed bias?

        For most of them the answer is no, and that is the whole reason
        the sizing problem splits into 'solve the bias, then solve the
        width'. The edge correction breaks it for a few: fT is
        gm/(2*pi*C) with a numerator proportional to the count and a
        denominator affine in it, so it drifts about 1.4 % across the
        useful range, and Cgg_tot and Cdtot carry the end term directly.

        Without an edge table the answer is no for everything, and not
        because it is true: the table simply holds no information about
        the end term, so there is nothing to iterate towards. The 1.4 %
        and 6.6 % are then wrong and stay wrong, which is what the report
        says when no edge table is attached.
        """
        if self.edge is None:
            return False
        d = deps_of(key)
        if d is None:
            # not a single key -- try it as an expression before giving
            # up, so that a chart of 'GM/GDS*VSTAR' is not labelled
            # finger-dependent merely because it has an asterisk in it
            d = deps_of_expr(key, self.params)
        return True if d is None else bool(d & NG_LAW)

    # -- geometry -------------------------------------------------------
    def weff(self, w_drawn, ng=None):
        ng = self.ng if ng is None else ng
        return w_drawn - ng * self.dW

    def w_drawn(self, w_eff, ng=None):
        ng = self.ng if ng is None else ng
        return w_eff + ng * self.dW

    # -- storage --------------------------------------------------------
    def _tab(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(os.path.join(self.path, name + ".npy"),
                                        mmap_mode="r")
        return self._cache[name]

    def _flat(self, ig, idd, ib):
        nG, nD = len(self.VGS), len(self.VDS)
        return ig + idd * nG + ib * nG * nD + self.OS[idd, ib]

    # Set False to fall back to pure multilinear everywhere.
    CUBIC = True

    # Parameters whose dependence on VDS is EXPONENTIAL rather than a
    # power law, so the VDS slices are combined in the log domain for
    # these and in the linear domain for everything else. Only gds is
    # listed, because gds is the only one of these the probe measures;
    # iavl and ijd are probably in the same family, and they are left out
    # until there is a measurement rather than a guess. See the long
    # comment at the end of raw_at for the trade-off this makes.
    VDS_LOG = frozenset({"gds"})

    # Exact node slopes along VGS for ID, with gm derived from that same
    # interpolant; see _id_gm_flat for what it does and why only VGS.
    #
    # ON, after being measured twice and refused once.
    #
    # Measured against SPICE on 720 off-grid probe points of the real
    # table, p90 of |error| by inversion region:
    #
    #                  weak    moderate   strong     all
    #       ID        1.94x     2.01x     1.00x     1.76x
    #       gm        1.81x     1.41x     0.24x     1.23x
    #       gm/ID     1.27x     0.81x     1.14x     1.15x
    #       fT        1.00x     1.00x     1.00x     1.00x   <- control
    #
    # fT is built from the capacitances, which this does not touch, so
    # its 1.00x in every region is what says the measurement itself is
    # sound rather than a reshuffle.
    #
    # The gain is where the argument says it should be and nowhere else.
    # ln(ID) is near-straight in VGS only while d(ln ID)/dVGS = gm/ID is
    # near-constant, which is weak inversion; in strong inversion ID
    # goes as a power of the overdrive instead, the supplied slope stops
    # helping, and the order penalty is all that is left. That penalty
    # is real: gm is the derivative of an interpolant, one order lower
    # than a gm read from its own table, and in strong inversion it
    # measures 4x WORSE -- 0.0041 % becomes 0.0170 %.
    #
    # Turned on anyway, and the reason is absolute size rather than
    # ratio. 0.0170 % sits 60x below this table's own worst point
    # (1.501 % of ID, at L = 0.613 um, inside the model's reverse
    # short-channel hump), so the regression is in the one quantity
    # that had accuracy to spare. What is bought with it is 2x on ID
    # across weak and moderate inversion -- ID is what the width solve
    # inverts -- plus half the query cost: 31.5 ms to 15.3 ms over 4000
    # points, because supplied slopes need two VGS nodes instead of
    # four and one gather then serves ids and gm together.
    #
    # Worth recording that a synthetic fixture said the opposite. Its L
    # grid has 10 lengths against the real 37, so L error swamped
    # everything and the comparison came back 1.0x with gm/ID at 0.54x.
    # The fixture was not wrong about its own device; it was the wrong
    # device to ask. Flip this to False and re-run check_probe.py to
    # re-measure on any new table rather than inheriting this verdict.
    #
    # -------------------------------------------------------------- #
    # RE-MEASURED ON EIGHT TABLES, AND TURNED OFF.
    # -------------------------------------------------------------- #
    # The paragraphs above are all correct and all measured on NMOS.
    # check_probe.py was then run on the two tables they were written
    # from plus six sg13_hv_pmos tables at -40 and 125 C. The n tables
    # reproduced the verdict exactly:
    #
    #   lut_nmos_hv_tt_27   ID  weak 1.94x  moderate 2.01x  strong 1.00x
    #   lut_nmos_lv_tt_27   ID  weak 2.15x  moderate 2.50x  strong 1.30x
    #   worst supplied-slope error on either table: 0.0284 %
    #
    # The p tables did not. Four of the six blow up:
    #
    #   hv_p_tt_125   gm    moderate    29.4453 %
    #   hv_p_tt_m40   gm    strong     100.0000 %
    #   hv_p_ff_125   gm    strong     102.1182 %
    #   hv_p_ff_m40   gm/ID strong     150.8425 %
    #
    # with np.exp overflowing in the log-space Hermite. The remaining
    # two (ss at both temperatures) are clean, so it is not every p
    # table -- which makes it worse, not better: there is no rule here
    # that a reader could apply without running the probe.
    #
    # Against that, the ESTIMATED path's worst error across all eight
    # tables, every region, every quantity, is 1.2222 %, and almost all
    # of it is under 0.07 %.
    #
    # So the trade is: 2x on errors that are already 0.03 %, against
    # 150 % on half the p tables. 0.0325 % becoming 0.0168 % changes no
    # design decision; 150 % changes every one. Off.
    #
    # The cost of turning it off is the ID/gm consistency the supplied
    # path gives by construction, since gm is then interpolated from
    # its own table rather than differentiated out of the ID
    # interpolant. That inconsistency measures at the 0.01 % level,
    # below the noise of anything else in this project.
    #
    # The deeper point is the one this project keeps relearning: this
    # was a numerical decision measured on ONE device and applied to
    # seventy-two tables. Same disease as rg, dW and the five hv
    # constants, in the engine instead of in a report. The instruction
    # above -- re-measure, do not inherit -- was right, and it took a
    # pmos table to collect on it.
    SLOPE = False

    @staticmethod
    def _hermite2(h, f):
        """
        Two-node Hermite on one cell, with the node slopes supplied from
        outside instead of estimated from neighbours.

        Returns (value weights, slope weights, and the derivatives of
        both with respect to x), each a pair for nodes (i, i+1):

            S      = wv0*Y0 + wv1*Y1 + ws0*M0 + ws1*M1
            dS/dx  = dv0*Y0 + dv1*Y1 + ds0*M0 + ds1*M1

        where Y is the interpolated field and M its exact slope at the
        node. This is the whole point of the scheme: for ln(ID) along
        VGS the slope IS gm/ID, which is tabulated, so nothing has to be
        estimated and the stencil needs two nodes instead of four.

        Checks worth keeping in mind: at f = 0 the value reduces to Y0
        and the derivative to M0 exactly, and at f = 1 to Y1 and M1. So
        the interpolant passes through the stored values AND the stored
        slopes at every node, which is what makes gm derived from it
        agree with the gm table at the nodes rather than merely near
        them.
        """
        f2, f3 = f * f, f * f * f
        wv0, wv1 = 2 * f3 - 3 * f2 + 1, -2 * f3 + 3 * f2
        ws0, ws1 = h * (f3 - 2 * f2 + f), h * (f3 - f2)
        dv = (6 * f2 - 6 * f) / h
        return ((wv0, wv1), (ws0, ws1),
                (dv, -dv), (3 * f2 - 4 * f + 1, 3 * f2 - 2 * f))

    @staticmethod
    def _outside(grid, x):
        """
        Points beyond either end of this axis.

        _brk CLAMPS: searchsorted is clipped into the last cell and the
        fraction into [0, 1], so a query past the top node comes back as
        the top node's value. That is right for the arithmetic -- the
        stencil has to land somewhere -- and wrong as an answer, and the
        incomplete-grid mask never caught it because that mask only knows
        about the SOA clip on VGS and VDS. Measured on a table whose VSB
        axis stopped at 0.9 V: a query at VSB = 2.5 V returned the
        0.9 V current, to the last digit, with nothing to say so. The
        same held for L at 10 um on an axis ending at 3 um, and for a
        negative VGS.

        This file opens by promising that a query it cannot answer comes
        back as NaN rather than as a clipped neighbour. It kept that
        promise for the clip and broke it at the ends.
        """
        lo, hi = float(grid[0]), float(grid[-1])
        tol = 1e-9 * max(abs(lo), abs(hi), 1.0)
        return (x < lo - tol) | (x > hi + tol)

    @staticmethod
    def _brk(grid, x, log=False):
        i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
        if log:
            f = (np.log(np.maximum(x, 1e-30)) - np.log(grid[i])) / \
                (np.log(grid[i + 1]) - np.log(grid[i]))
        else:
            f = (x - grid[i]) / (grid[i + 1] - grid[i])
        return i, np.clip(f, 0.0, 1.0)

    @staticmethod
    def _quadw(xa, xb, xc, t):
        """
        Weights on y(xa), y(xb), y(xc) that give dy/dx at the point t,
        from the parabola through the three nodes. Second order accurate
        on ANY spacing, and it does not care whether t sits inside the
        triple or at its edge -- which is exactly what lets the same
        formula serve the interior and the two ends.
        """
        wa = ((t - xb) + (t - xc)) / ((xa - xb) * (xa - xc))
        wb = ((t - xa) + (t - xc)) / ((xb - xa) * (xb - xc))
        wc = ((t - xa) + (t - xb)) / ((xc - xa) * (xc - xb))
        return wa, wb, wc

    @classmethod
    def _stencil(cls, x, i, f, cubic, nval=None):
        """
        Stencil indices and weights for one axis.

        linear : indices (i, i+1),              error O(h^2)
        cubic  : indices (i-1, i, i+1, i+2),    error O(h^4)

        The cubic is a Hermite, and the two node slopes it needs are
        taken from the parabola through three neighbouring nodes. That
        is valid on ANY spacing, which this axis needs: the L grid is
        deliberately non-uniform (50 nm steps near Lmin, 1 um at the
        top, a 3.0x spread in log spacing). On a uniform axis the
        interior reduces exactly to Catmull-Rom, so VGS and VSB get the
        textbook scheme for free.

        At the two ENDS the three-node support is taken ONE-SIDED
        (nodes 0,1,2 or n-3,n-2,n-1) instead of clamping the index.
        Clamping silently degrades the end slope to a one-sided FIRST
        difference, which is only first order; the one-sided parabola
        stays second order. Measured on the real L grid against a
        1/L x exp() x VTH-rolloff model, inside the first cell
        (L = 0.45 - 0.50 um):

            linear            max error  1.393 %
            clamped slope     max error  0.885 %      1.6x
            one-sided parab.  max error  0.057 %     24.5x

        and that first cell is precisely where every worst-case point
        of the accuracy probe sat.

        nval gives, PER QUERY POINT, how many nodes of this axis really
        exist. The VGS axis is shorter at high VSB (the SOA clip), so a
        stencil reaching i+2 blindly would ask for a column that was
        never simulated and the query would come back NaN. With nval the
        stencil shrinks instead: it falls back to its own one-sided form
        at the clip edge, and to plain linear where only two nodes are
        left. Measured on the synthetic table that recovered 100 % of
        the coverage the wider stencil had cost.

        Returns index arrays, not offsets, because the end handling
        makes the offsets position-dependent. Slots may repeat an index
        at the ends; the gather sums the weights, so that is harmless.
        """
        n = len(x)
        i1 = i + 1
        if not cubic or n < 3:
            return [i, i1], [1 - f, f]
        if nval is None:
            nval = n
        nv = np.minimum(nval, n)
        # Points with nv < 3 drop to linear at the bottom of this
        # routine, but the geometry above still has to be evaluated for
        # them, so widen it to a non-degenerate triple first. Otherwise
        # two of the three support nodes coincide and _quadw divides by
        # zero. Their weights are overwritten, and the extra slot they
        # pull in carries weight 0, which the gather skips.
        nvg = np.maximum(nv, 3)

        im = np.maximum(i - 1, 0)
        ip = np.minimum(i + 2, nvg - 1)
        slots = [im, i, i1, ip]
        w = [np.zeros_like(f) for _ in range(4)]

        t = f
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        h = x[i1] - x[i]

        w[1] += h00
        w[2] += h01

        # slope at the LEFT node x[i]:
        #   interior   -> three-node support (im, i, i1)  = slots 0,1,2
        #   i == 0     -> one-sided support  (0, 1, 2)    = slots 1,2,3
        cl = i > 0
        wa, wb, wc = cls._quadw(np.where(cl, x[im], x[i]),
                                np.where(cl, x[i], x[i1]),
                                np.where(cl, x[i1], x[ip]), x[i])
        k = h10 * h
        w[0] += k * np.where(cl, wa, 0.0)
        w[1] += k * np.where(cl, wb, wa)
        w[2] += k * np.where(cl, wc, wb)
        w[3] += k * np.where(cl, 0.0, wc)

        # slope at the RIGHT node x[i+1]:
        #   interior   -> three-node support (i, i1, ip)        = slots 1,2,3
        #   i1 == n-1  -> one-sided support  (n-3, n-2, n-1)    = slots 0,1,2
        cr = i < nvg - 2
        wa, wb, wc = cls._quadw(np.where(cr, x[i], x[im]),
                                np.where(cr, x[i1], x[i]),
                                np.where(cr, x[ip], x[i1]), x[i1])
        k = h11 * h
        w[0] += k * np.where(cr, 0.0, wa)
        w[1] += k * np.where(cr, wa, wb)
        w[2] += k * np.where(cr, wb, wc)
        w[3] += k * np.where(cr, wc, 0.0)

        # fewer than three nodes left on this axis -> nothing to fit a
        # parabola to. Drop those points, and only those, to linear.
        thin = nv < 3
        if np.any(thin):
            w[0] = np.where(thin, 0.0, w[0])
            w[1] = np.where(thin, 1 - f, w[1])
            w[2] = np.where(thin, f, w[2])
            w[3] = np.where(thin, 0.0, w[3])

        return slots, w

    # Largest number of query points handled in one tensor. The gather
    # builds (corners x points) arrays, so this caps peak memory at a few
    # tens of MB; bigger queries are chunked, which costs nothing because
    # the per-call overhead is already amortised at this width.
    CHUNK = 4096

    def raw_at(self, name, L, VGS, VDS, VSB):
        """One stored parameter at W_ref, 4-D interpolated. NaN off-grid."""
        if self.SLOPE and name in ("ids", "gm"):
            return self.id_gm_at(L, VGS, VDS, VSB)[name == "gm"]
        L, VGS, VDS, VSB = np.broadcast_arrays(
            *[np.asarray(v, float) for v in (L, VGS, VDS, VSB)])
        shape = L.shape
        L, VGS, VDS, VSB = (a.ravel() for a in (L, VGS, VDS, VSB))
        n = L.size
        if n > self.CHUNK:
            parts = [self._raw_flat(name, L[k:k + self.CHUNK],
                                    VGS[k:k + self.CHUNK],
                                    VDS[k:k + self.CHUNK],
                                    VSB[k:k + self.CHUNK])
                     for k in range(0, n, self.CHUNK)]
            return np.concatenate(parts).reshape(shape)
        return self._raw_flat(name, L, VGS, VDS, VSB).reshape(shape)

    def id_gm_at(self, L, VGS, VDS, VSB):
        """
        ids and gm together, with gm DERIVED from the ids interpolant.

        Returns (ids, gm). One gather serves both, so asking for gm/ID
        costs a single pass instead of two.
        """
        L, VGS, VDS, VSB = np.broadcast_arrays(
            *[np.asarray(v, float) for v in (L, VGS, VDS, VSB)])
        shape = L.shape
        L, VGS, VDS, VSB = (a.ravel() for a in (L, VGS, VDS, VSB))
        n = L.size
        if n > self.CHUNK:
            parts = [self._id_gm_flat(L[k:k + self.CHUNK],
                                      VGS[k:k + self.CHUNK],
                                      VDS[k:k + self.CHUNK],
                                      VSB[k:k + self.CHUNK])
                     for k in range(0, n, self.CHUNK)]
            return (np.concatenate([p[0] for p in parts]).reshape(shape),
                    np.concatenate([p[1] for p in parts]).reshape(shape))
        a, b = self._id_gm_flat(L, VGS, VDS, VSB)
        return a.reshape(shape), b.reshape(shape)

    def _id_gm_flat(self, L, VGS, VDS, VSB):
        """
        ID with EXACT node slopes along VGS, and gm as the derivative of
        that same interpolant.

        Why only VGS. The exact slope of ln(ID) along VGS is gm/ID and
        along VDS is gds/ID and along VSB is -gmb/ID, all three
        tabulated, so in principle all three axes could use supplied
        slopes. They cannot, and the reason is not laziness: a
        tensor-product Hermite with supplied slopes on TWO axes also
        needs the MIXED partial at each corner -- d(gm/ID)/dVDS and the
        like -- which is not tabulated and cannot be had without
        differentiating a table, i.e. without going back to estimates.
        With supplied slopes on ONE axis there is no mixed term: the VGS
        operator is applied first at every (L, VDS, VSB) node of the
        stencil, and the remaining axes then interpolate its RESULT as
        plain values. So one axis is the exact stopping point, and VGS is
        the one worth having -- in weak inversion ln(ID) is nearly
        straight in VGS, since d(ln ID)/dVGS = gm/ID is nearly constant
        there.

        Why derive gm rather than read it. Read independently, ids and
        gm are two interpolants that agree at the nodes and drift
        between them, so gm/ID carries the SUM of two unrelated errors
        and nothing cancels. Derived, gm = ID * dS/dVGS by construction,
        so gm/ID is exactly the interpolated slope -- and at a node it
        is exactly the stored gm/ID, because the Hermite reproduces the
        supplied slope there. Circuits that cancel to first order, a
        bandgap above all, are sensitive to exactly that consistency
        and not to the per-parameter error.

        A side benefit at the SOA clip: two nodes are needed where four
        were, so queries near the clip edge that the wider stencil had
        to refuse now resolve.
        """
        off = (self._outside(self.L, L) | self._outside(self.VGS, VGS)
               | self._outside(self.VDS, VDS) | self._outside(self.VSB, VSB))

        iL, fL = self._brk(self.L, L, log=True)
        ig, fg = self._brk(self.VGS, VGS)
        idd, fd = self._brk(self.VDS, VDS)
        ib, fb = self._brk(self.VSB, VSB)

        cub = self.CUBIC
        IL, WL = self._stencil(self._logL, iL, fL, cub)
        IB, WB = self._stencil(self.VSB, ib, fb, cub)
        nval = np.minimum.reduce([self.n_b[bi] for bi in IB])
        ID_, WD = self._stencil(self.VDS, idd, fd, cub, nval=nval)

        # the VGS cell itself, and the four weight pairs on it
        hg = self.VGS[ig + 1] - self.VGS[ig]
        (wv, ws, dv, ds) = self._hermite2(hg, fg)
        IG = [ig, ig + 1]

        n = L.size
        kL, kD, kB, kG = len(IL), len(ID_), len(IB), 2

        def col(idx, k, ax):
            sh = [1, 1, 1, 1, n]
            sh[ax] = k
            return np.broadcast_to(np.stack(idx), (k, n)).reshape(sh)

        li, gi, di, bi = (col(IL, kL, 0), col(IG, kG, 1),
                          col(ID_, kD, 2), col(IB, kB, 3))
        wl, wd, wb = col(WL, kL, 0), col(WD, kD, 2), col(WB, kB, 3)
        wvg, wsg = col(wv, kG, 1), col(ws, kG, 1)
        dvg, dsg = col(dv, kG, 1), col(ds, kG, 1)

        nG, nD = len(self.VGS), len(self.VDS)
        nb = self.n_b[bi]
        inside = (gi < nb) & (di < nb)
        flat = gi + di * nG + bi * nG * nD + self.OS[di, bi]
        safe = np.where(inside, flat, 0)
        tid, tgm = self._tab("ids"), self._tab("gm")
        # float64 for the same reason as in _raw_flat: float32 storage is
        # fine, float32 logs are not.
        A = np.where(inside, tid[li, safe], 1.0).astype(np.float64,
                                                        copy=False)
        G = np.where(inside, tgm[li, safe], 0.0).astype(np.float64,
                                                        copy=False)

        pos = A > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = np.log(np.where(pos, A, 1.0))        # ln ID at the nodes
            M = np.where(pos, G / np.where(pos, A, 1.0), 0.0)   # gm/ID

        # VGS first, then L and VSB as plain value weights on the result,
        # all of it in the log domain -- the same domain _raw_flat uses
        # for these three axes, and for the same reason.
        w3 = wl * wb
        S = np.sum(w3 * (wvg * Y + wsg * M), axis=(0, 1, 3))   # (kD, n)
        dS = np.sum(w3 * (dvg * Y + dsg * M), axis=(0, 1, 3))

        # "is this stencil slot actually used" -- on ABSOLUTE weights.
        # wv1 and ws1 have opposite signs, so their sum can vanish while
        # both are non-zero; it cannot happen for a cell narrower than
        # 1 V (the root sits at f = (3-h)/(2-h) > 1), but a mask that is
        # only correct for the grids we happen to use is a trap left for
        # later.
        used = (np.abs(wl) * np.abs(wb)
                * (np.abs(wvg) + np.abs(wsg)) * np.abs(wd)) != 0
        ok = np.all(inside | ~used, axis=(0, 1, 2, 3))
        # The log domain needs every used current strictly positive. Deep
        # subthreshold in float32 storage underflows to exactly 0, and
        # _raw_flat handles that by falling back to its linear combine --
        # so this path must too, or a query at VGS = 0 would start
        # returning NaN where it used to return a number, and the
        # sizer's low-end scan would lose its bracket.
        fall = ~np.all(pos | ~used, axis=(0, 1, 2, 3))

        wd2 = wd.reshape(kD, n)
        E = np.exp(S)
        ids = np.sum(wd2 * E, axis=0)
        # gm = d(ids)/dVGS of the expression actually evaluated, VDS
        # combine included, so the two stay consistent.
        gm = np.sum(wd2 * E * dS, axis=0)
        if fall.any():
            ai = self._raw_flat("ids", L, VGS, VDS, VSB)
            ag = self._raw_flat("gm", L, VGS, VDS, VSB)
            ids = np.where(fall, ai, ids)
            gm = np.where(fall, ag, gm)
        bad = ~(ok & ~off)
        return (np.where(bad, np.nan, ids), np.where(bad, np.nan, gm))

    def _raw_flat(self, name, L, VGS, VDS, VSB):
        """
        The gather, as ONE tensor instead of a Python loop over corners.

        The cubic stencils touch 4 nodes per axis, so 4^4 = 256 corners.
        Walking those in Python cost about 4.2 ms per call whatever the
        number of query points -- the work was numpy call overhead, not
        arithmetic, so a 1-point query cost the same as a 4000-point one.
        Building the whole (256, points) index tensor and gathering once
        turns ~1000 tiny numpy calls into about 15 large ones.
        """
        off = (self._outside(self.L, L) | self._outside(self.VGS, VGS)
               | self._outside(self.VDS, VDS) | self._outside(self.VSB, VSB))

        iL, fL = self._brk(self.L, L, log=True)
        ig, fg = self._brk(self.VGS, VGS)
        idd, fd = self._brk(self.VDS, VDS)
        ib, fb = self._brk(self.VSB, VSB)

        tab = self._tab(name)
        use_log = name not in NO_LOG
        cub = self.CUBIC

        IL, WL = self._stencil(self._logL, iL, fL, cub)
        IB, WB = self._stencil(self.VSB, ib, fb, cub)
        # Every VSB row the VSB stencil touches must supply the same
        # (VGS, VDS) columns, so the VGS and VDS stencils are bounded by
        # the SHORTEST of those rows. Narrowing the fit keeps it inside
        # data that exists; it never moves the evaluation point.
        nval = np.minimum.reduce([self.n_b[bi] for bi in IB])
        IG, WG = self._stencil(self.VGS, ig, fg, cub, nval=nval)
        ID_, WD = self._stencil(self.VDS, idd, fd, cub, nval=nval)

        n = L.size
        kL, kG, kD, kB = len(IL), len(IG), len(ID_), len(IB)

        def col(idx, k, ax):
            sh = [1, 1, 1, 1, n]
            sh[ax] = k
            return np.broadcast_to(np.stack(idx), (k, n)).reshape(sh)

        li = col(IL, kL, 0)
        gi = col(IG, kG, 1)
        di = col(ID_, kD, 2)
        bi = col(IB, kB, 3)
        wl = col(WL, kL, 0)
        wg = col(WG, kG, 1)
        wd = col(WD, kD, 2)
        wb = col(WB, kB, 3)

        nG, nD = len(self.VGS), len(self.VDS)
        nb = self.n_b[bi]
        inside = (gi < nb) & (di < nb)
        flat = gi + di * nG + bi * nG * nD + self.OS[di, bi]
        # float64 ON PURPOSE. The table is stored float32, which is right
        # for 0.93 GiB of it, but np.where against a float32 array keeps
        # float32, and then np.log runs in single precision: a constant
        # field came back 47.000005754753 instead of 47.0, an error of
        # exactly float32 epsilon. The DATA may be float32; the
        # ARITHMETIC on top of it must not be.
        V = np.where(inside, tab[li, np.where(inside, flat, 0)],
                     1.0).astype(np.float64, copy=False)

        w3 = wl * wg * wb                      # (kL,kG,1,kB,n)
        # a stencil point only has to exist if it is actually used
        ok = np.all(inside | ((w3 * wd) == 0), axis=(0, 1, 2, 3))

        # ------------------------------------------------------------------
        # VDS is combined SEPARATELY from the other three axes, because the
        # domain that linearises the physics is not the same on all four.
        #
        # The log domain is exact for an exponential and wrong for a power
        # law. Along VGS and VSB the weak-inversion current IS an
        # exponential, which is why the log domain belongs there. Along
        # VDS, near VDS = 0, ID is proportional to VDS: a power law. A
        # geometric mean of two currents 25 mV apart then lands at
        # sqrt(0.050*0.075) = 0.0612 instead of 0.0625, which is the 2 %
        # the probe kept reporting in deep triode. Measured on the real
        # VDS grid against a textbook triode law, over the first cells:
        # linear domain 0.51 %, log domain 5.59 %.
        #
        # Splitting also removes an all-or-nothing failure: one single 4-D
        # log combine needed every corner positive, so the zero at
        # VDS = 0 dropped the whole point to plain multilinear and threw
        # away the log benefit on VGS and VSB too. Now only the VDS = 0
        # slice falls back.
        # ------------------------------------------------------------------
        ax = (0, 1, 3)
        lin_sl = np.sum(w3 * V, axis=ax)                      # (kD, n)
        sl = lin_sl
        if use_log:
            pos = np.all(V > 0, axis=ax)
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.exp(np.sum(
                    w3 * np.log(np.where(V > 0, V, 1.0)), axis=ax))
            sl = np.where(pos & np.isfinite(lg), lg, lin_sl)

        wd2 = wd.reshape(kD, n)
        out = np.sum(wd2 * sl, axis=0)
        if use_log and name in self.VDS_LOG:
            posd = np.all(sl > 0, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lgd = np.exp(np.sum(
                    wd2 * np.log(np.where(sl > 0, sl, 1.0)), axis=0))
            out = np.where(posd & np.isfinite(lgd), lgd, out)
        return np.where(ok & ~off, out, np.nan)

    # -- public ---------------------------------------------------------
    def eval(self, what, L, VGS, VDS=0.0, VSB=0.0, W=None, ng=None,
             parity=True):
        """
        Public entry: _eval, with numpy's divide warnings silenced.

        A ratio such as GM/ID at VGS = 0 or VDS = 0 is 0/0: the device
        carries no current there and the quantity has no value. NaN is
        the correct answer and every caller already treats it as 'no
        point here' (a gap in a curve, a refused cursor). The warning
        numpy prints for it reports nothing wrong, and in the window it
        scrolled the terminal on every sweep that starts at zero volts.
        Only the WARNING is suppressed; the NaN is returned as before.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._eval(what, L, VGS, VDS, VSB, W, ng, parity)

    def _eval(self, what, L, VGS, VDS=0.0, VSB=0.0, W=None, ng=None,
              parity=True):
        """
        `what` is a key of DERIVED, a raw parameter name, or an expression
        built from them, e.g. "GM/ID*FT".  W is the DRAWN width; omit it
        to get the reference device.

        parity=False asks for the junctions' continuous odd-branch form
        instead of the true parity law. It is for an iteration that is
        still looking for a finger count; a device that exists has an
        integer count and wants the default. See finger_regions.
        """
        ng = self.ng if ng is None else ng
        scale = 1.0 if W is None else self.weff(W, ng) / self.W_ref_eff

        # With the edge table attached, the affine parameters are
        # corrected here, where ng is known. eval() will multiply what
        # _Lazy hands back by scale, so what is handed back is the
        # corrected value DIVIDED by scale -- which is exact, because
        # the widths this tool produces are always ng whole reference
        # fingers and so scale is exactly ng/ng_ref. That condition is
        # checked rather than assumed: a width that is not a whole
        # number of fingers has no defined end term.
        edge = self.edge
        correct = edge is not None and W is not None
        if correct:
            # W and ng may be ARRAYS (one device per sweep point: the
            # sizing assistant's ID-driven sweeps size a different width
            # at every x). Nothing below needs a scalar; this check was
            # the only thing that assumed one.
            want = np.asarray(ng, float) / self.ng
            bad = np.abs(scale - want) > 1e-6 * np.maximum(want, 1.0)
            if np.any(bad):
                _i = int(np.argmax(bad))
                _f = lambda v: float(np.asarray(v, float).ravel()[
                    _i if np.size(v) > 1 else 0])
                raise ValueError(
                    f"the edge correction needs a width of whole "
                    f"reference fingers: ng = {_f(ng):g} gives scale "
                    f"{_f(want):.6f} but W = {_f(W):.6g} gives "
                    f"{_f(scale):.6f}. "
                    f"Size with gmid_size, which realises W as a finger "
                    f"count.")

        class _Lazy(dict):
            """
            Raw parameters, fetched once each and only if asked for.

            The affine correction used to be computed EAGERLY here, all
            five parameters on every call, which is two extra 4-D gathers
            apiece. A gm/ID constraint needs none of them and an fT
            constraint needs two, so nine or ten gathers out of ten were
            thrown away -- and the sizer evaluates a constraint some fifty
            times per solve. Measured on the synthetic table, doing it on
            demand took a solve that depends on the finger count from
            6.3 s to 1.1 s and changed no number anywhere.
            """
            def __missing__(s, k):
                if k == "kt4":
                    s[k] = 4.0 * 1.380649e-23 * self.temp_k
                    return s[k]
                if self.SLOPE and k in ("ids", "gm"):
                    # both from one pass: a gm/ID constraint then costs
                    # one gather instead of two
                    a, g = self.id_gm_at(L, VGS, VDS, VSB)
                    s["ids"], s["gm"] = a, g
                    return s[k]
                if correct and k in AFFINE:
                    p5 = self.raw_at(k, L, VGS, VDS, VSB)
                    p1 = edge.raw_at(k, L, VGS, VDS, VSB)
                    dn = self.ng - edge.ng
                    c = (p5 - p1) / dn
                    e0 = (self.ng * p1 - edge.ng * p5) / dn
                    s[k] = (c * ng + e0) / scale
                elif correct and k in JUNCTION:
                    # two region counts, two tables, one 2x2 solve. Doing
                    # it generally rather than for the 5-and-1 pair, so a
                    # table characterised at another finger count does
                    # not silently fall back to the wrong arithmetic.
                    term = JUNCTION[k]
                    pm = self.raw_at(k, L, VGS, VDS, VSB)
                    pe = edge.raw_at(k, L, VGS, VDS, VSB)
                    im, em = finger_regions(self.ng, term)
                    ie, ee = finger_regions(edge.ng, term)
                    det = im * ee - ie * em
                    if abs(float(det)) < 1e-12:
                        raise ValueError(
                            f"the two tables, at ng = {self.ng} and "
                            f"ng = {edge.ng}, give the same {term} "
                            f"region counts, so the shared and end "
                            f"terms of {k!r} cannot be separated. "
                            f"Characterise the edge table at a finger "
                            f"count of the other parity.")
                    a = (pm * ee - pe * em) / det
                    b = (im * pe - ie * pm) / det
                    i_n, e_n = finger_regions(ng, term, parity)
                    s[k] = (i_n * a + e_n * b) / scale
                else:
                    s[k] = self.raw_at(k, L, VGS, VDS, VSB)
                return s[k]

        lazy = _Lazy()

        def fetch(nm):
            key = nm.upper()
            if key == "L":
                return np.asarray(L, float)
            if key == "W":
                return np.broadcast_to(np.asarray(
                    self.W_nom if W is None else W, float),
                    np.shape(L)).astype(float)
            if key == "WEFF":
                return np.broadcast_to(np.asarray(self.weff(
                    self.W_nom if W is None else W, ng), float),
                    np.shape(L)).astype(float)
            if key == "AREA":
                return np.asarray(L, float) * (self.W_nom if W is None
                                               else W)
            if key == "VGS":
                return np.asarray(VGS, float)
            if key == "VDS":
                return np.asarray(VDS, float)
            if key == "VSB":
                return np.asarray(VSB, float)
            if key == "TEMPK":
                # so a chart expression never has to type a temperature
                return np.full(np.shape(L), self.temp_k, float)
            # fug is the model's own unity-gain frequency, a RATIO of a
            # proportional numerator to an affine denominator, so it is
            # neither proportional nor affine and cannot be corrected in
            # place. Once the capacitances are corrected it is rebuilt
            # from them instead. Verified on this PDK to equal
            # gm/(2.pi.(cgg+cgsol+cgdol)) to 0.0002 %.
            if key in ("FT", "FUG") and correct:
                return lazy["gm"] / (2.0 * np.pi * (lazy["cgg"]
                                                    + lazy["cgsol"]
                                                    + lazy["cgdol"]))
            if key in PER_W:
                fn, _ = DERIVED[key]
                return fn(lazy) / self.W_ref_eff
            if key in DERIVED:
                fn, p = DERIVED[key]
                return fn(lazy) * (scale ** p)
            # Raw parameters are stored lower case. A chart expression
            # is written in the same breath as the DERIVED keys, which
            # are upper case, so "SID/(4*k*T*GM)" would die on SID while
            # "sid/(4*k*T*GM)" worked -- a trap with no purpose. Accept
            # either spelling.
            nm = nm if (nm in self.params or nm in RAW_P) else key.lower()
            if nm in self.params or nm in RAW_P:
                if nm not in RAW_P:
                    raise KeyError(
                        f"{nm!r} is a stored parameter but its W-scaling "
                        f"is not recorded in RAW_P, so eval() will not "
                        f"guess it. Add it there, and measure it with "
                        f"make_verify.py rather than inferring it.")
                # through _Lazy, not raw_at, so that asking for 'cgsol'
                # and asking for 'CGSOL' cannot give different answers:
                # the lower-case path used to skip the edge correction.
                return lazy[nm] * (scale ** RAW_P[nm])
            raise KeyError(nm)

        s = str(what)
        if s.upper() in DERIVED or s.upper() in ("L", "W", "WEFF", "AREA",
                                                 "VGS", "VDS", "VSB") \
                or s in self.params:
            return fetch(s)

        env = dict(ENV)
        # TOKENISING, done properly on the third attempt.
        #
        # The first version put '/' in the character class so that
        # "GM/ID" and "GM/GDS" would arrive as single DERIVED keys. That
        # works for them and breaks everything else that contains a
        # slash, and each break needed its own patch: a ratio that is
        # not itself a key ("CGG_TOT/W") arrived whole and failed; a
        # dividing slash at the end ("SID/(...)") was swallowed; a
        # numeric divisor ("FT/1e9") was eaten because digits and 'e'
        # are in the class; and "ID/1e-6" defeated it altogether,
        # because the '-' of the exponent stops the match mid-literal.
        #
        # Patching the class further is the wrong direction. There is a
        # FINITE set of keys that contain a slash -- they are the ones
        # in DERIVED and PER_W -- so match those explicitly, longest
        # first, and let every other token be a plain identifier with no
        # slash in it. Then no slash is ever ambiguous: it is either
        # inside a key this matched whole, or it is division.
        toks = tokens(s)
        safe = s
        for i, t in enumerate(toks):
            try:
                env[f"_v{i}"] = fetch(t)
            except KeyError:
                continue        # a function or constant already in env
            safe = safe.replace(t, f"_v{i}")
        return eval(safe, {"__builtins__": {}}, env)          # noqa: S307


if __name__ == "__main__":
    import sys
    lut = LUT(sys.argv[1] if len(sys.argv) > 1 else "lut_nmos_hv_tt_27")
    print(f"{lut.device}  W_ref drawn={lut.W_nom*1e6:g}u "
          f"eff={lut.W_ref_eff*1e6:g}u  ng={lut.ng}  dW={lut.dW*1e6:g}u/finger")
    print(f"L {len(lut.L)}  VGS {len(lut.VGS)}  VDS {len(lut.VDS)}  "
          f"VSB {len(lut.VSB)}  params {len(lut.params)}")
    print()
    print("  gm/ID sweep at L=1u, VDS=1.65, VSB=0")
    for vg in [0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.6, 3.2]:
        q = dict(L=1e-6, VGS=vg, VDS=1.65, VSB=0.0)
        print(f"    VGS={vg:4.2f}  gm/ID={float(lut.eval('GM/ID', **q)):6.2f}"
              f"  Vstar={float(lut.eval('VSTAR', **q))*1e3:7.1f}mV"
              f"  gm/gds={float(lut.eval('GM/GDS', **q)):7.1f}"
              f"  fT={float(lut.eval('FT', **q))/1e9:6.2f}GHz")
