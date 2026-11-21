#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | SIZING ASSISTANT
=============================================================================
Turns specifications into a device.

    from gmid_size import Sizer
    s = Sizer("lut_nmos_hv_tt_27")
    r = s.solve(VDS=0.9, VSB=0.0, **{"GM/ID": 15, "FT": 10e9, "ID": 50e-6})
    print(r)

or from the shell:

    python3 gmid_size.py lut_nmos_hv_tt_27 VDS=0.9 VSB=0 GM/ID=15 FT=10G ID=50u

-----------------------------------------------------------------------------
THE CONTRACT
-----------------------------------------------------------------------------
A MOSFET has exactly five degrees of freedom: VGS, VDS, VSB, W, L.
Everything else -- ID, gm, fT, VTH -- is a consequence, not an input. So a
sizing tool solves for those five and nothing more.

VDS and VSB come from the circuit: they are the designer's bias plan, and
they are given. That leaves VGS and L as the only real unknowns, plus W.

W is different from the other four. Every stored parameter is either
proportional to W (ID, gm, gds, every capacitance), independent of it
(VTH, Vdsat, gm/ID, fT, gm/gds), or inversely proportional (ro, rg). So W
never moves the BIAS POINT; it only scales the strength. That is why the
table is stored at one reference width, and it is also why W is solved
LAST, in closed form, by a single division -- no iteration.

Three specifications are therefore needed, and the rule for which three is
not "two width-independent plus one width-setting", which is what it looks
like at first. Write the W-powers of the three knobs as a vector p. A
monomial K1^a1 K2^a2 K3^a3 has W-power a.p, so the width-independent
combinations are the KERNEL of p:

    at least one knob with p != 0  ->  kernel is 2-dimensional
                                   ->  exactly 2 bias constraints
                                       + 1 width constraint.  LEGAL

    all three knobs with p == 0    ->  kernel is 3-dimensional
                                   ->  3 bias constraints on 2 unknowns,
                                       and W unconstrained.  REFUSED

So (ID, gm, L) is legal, with the kernel spanned by gm/ID and L: two width
setters still carry only one width constraint between them, because their
ratio is width-independent. And (gm/ID, L, VGS) is refused, because VGS is
already determined by gm/ID and L, and nothing then fixes W.

-----------------------------------------------------------------------------
HOW THE TWO BIAS EQUATIONS ARE SOLVED
-----------------------------------------------------------------------------
Not with Newton. A Newton step needs a derivative of an interpolated table
and a starting guess, and it fails silently when either is bad. This walks
a contour instead:

  1. at each L on the stored grid, solve the FIRST constraint for VGS by
     bisection. Bisection cannot diverge and needs no initial guess; it
     needs only a sign change, which it verifies by scanning the whole
     valid VGS range at that L. That gives the curve VGS*(L).

  2. the SECOND constraint evaluated along that curve is a function of L
     alone. Bisect it in log L.

  3. W in closed form from the width-setting combination.

About 1100 table reads, ~8 ms.

-----------------------------------------------------------------------------
WHEN THERE IS NO SOLUTION
-----------------------------------------------------------------------------
It says so, and it reports the range the second constraint can actually
reach along the first constraint's curve. That is the feasible-region
information, and it is worth more than a clamped number that looks like an
answer. Nothing here ever returns a silently clipped value.
"""

import json
import math
import os
import sys

import numpy as np

from gmid_lut import DERIVED, LUT
from gmid_mm import Mismatch

MM_KNOBS = Mismatch.MM_KNOBS

# ---------------------------------------------------------------------------
# THE FINGER WIDTH IS NOT A FREE PARAMETER OF THIS TABLE.
#
# The table does not describe "a transistor of width W". It describes ONE
# FINGER of the width it was characterised at, and multiplies. Two
# measurements against the model settled this:
#
#   constant finger width, ng = 1, 2, 5, 10, 25, 50
#       ID/ng spread 0.0000 %, VTH identical to the last digit.
#       So at a fixed finger width the device IS the reference finger
#       repeated, exactly, over a 50x range.
#
#   constant total effective width 50 um, finger width 50 -> 1.94 um
#       the current DENSITY rises  +2.3 %, +3.6 %, +5.8 %, +15.9 %
#       while the capacitance density moves the other way by -3 to -5 %.
#       Same channel area: that is STI stress on the mobility plus an
#       edge capacitance, two different layout effects, and the curve
#       steepens sharply below 5 um -- right where the reference sits.
#
# The first version of this file held ng = 5 and stretched W continuously,
# which pushed the finger width from 1.94 um out to 49.6 um and produced
# ID errors up to 40 % against the model while gm/ID, VTH and Cgg all
# looked right to about 1 %. So W is realised by the NUMBER of fingers,
# and the finger width is read from the table, never chosen here.
#
# Residual, measured and not hidden: Cgg per finger is NOT constant with
# ng -- 7.384 fF at ng=1 down to 6.907 fF at ng=50, about +5.6 % to
# -1.2 % around the ng=5 reference, because the two outer edges are
# shared as fingers are added. fT inherits it. The table cannot see it.
# ---------------------------------------------------------------------------

# knobs that are the unknowns themselves, not stored parameters
GEOM = {"L", "VGS"}


def power_of(knob):
    """W-power of a knob: 0 width-independent, +1 ~W, -1 ~1/W."""
    k = knob.upper()
    if k in GEOM or k in ("VDS", "VSB"):
        return 0
    if k in ("W", "WEFF"):
        return 1
    if k == "AREA":
        return 1
    if k in MM_KNOBS:
        # Mismatch is NOT a power law in W: the threshold term goes as
        # W^-1/2, the width term as W^-1 and the length term as W^0. The
        # -0.5 here is only for ORDERING -- it says 'width-dependent,
        # but weaker than ID' so that ID wins the width-setting role
        # when both are asked for. _one_pass never raises W to it; it
        # solves the quadratic instead.
        return -0.5
    if k in DERIVED:
        return DERIVED[k][1]
    raise KeyError(f"unknown knob {knob!r}")


def fmt(v, unit=""):
    """engineering notation, because 7.3184e-06 is not a width"""
    if v is None or not np.isfinite(v):
        return "   --   "
    if v == 0:
        return f"0 {unit}"
    e = math.floor(math.log10(abs(v)) / 3) * 3
    e = max(-18, min(12, e))
    pre = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "u", -3: "m",
           0: "", 3: "k", 6: "M", 9: "G", 12: "T"}[e]
    return f"{v/10**e:.4g} {pre}{unit}".strip()


def parse_value(s):
    """accepts 10G, 50u, 1.2m, 15, 2.5e-6, and 2% for the mismatch knobs"""
    s = str(s).strip()
    # the mismatch report works in per cent, so a target written as 2%
    # means 2, not 0.02. Writing it bare works too; the sign is there to
    # be read, not to be converted.
    if s.endswith("%"):
        return float(s[:-1])
    mult = {"a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
            "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    if s and s[-1] in mult and not s[-1].isdigit():
        return float(s[:-1]) * mult[s[-1]]
    return float(s)


class NoSolution(Exception):
    pass


class Result(dict):
    """a solved device; prints as the report a designer wants to read"""

    REPORT = ["VGS", "VDS", "VSB", "VTH", "VOV", "VDSAT", "L", "W",
              "WEFF", "ID", "GM", "GDS", "GM/ID", "VSTAR", "GM/GDS",
              "RO", "VA", "CGG_TOT", "CGD", "CDTOT", "FT", "ID/W"]
    # Mismatch is computed, not looked up: it comes from the Pelgrom
    # coefficients and the DRAWN area, so it is kept in its own list and
    # never handed to eval(). Reported only -- see gmid_mm on why it is
    # not allowed to size anything until Monte Carlo has measured it.
    MM = ["SIGVT", "SIGBETA", "SIGL", "SIGID", "SIGID_PAIR",
          "SIGVOS_PAIR"]
    NOISE = ["GAMMA", "SID", "SVG", "SFL", "FKNEE", "SVG_1HZ"]
    UNIT = {"VGS": "V", "VDS": "V", "VSB": "V", "VTH": "V", "VOV": "V",
            "VDSAT": "V", "L": "m", "W": "m", "WEFF": "m", "ID": "A",
            "GM": "S", "GDS": "S", "RO": "ohm", "CGG_TOT": "F",
            "CGD": "F", "CDTOT": "F", "FT": "Hz", "VSTAR": "V",
            "VA": "V", "ID/W": "A/m",
            "SIGVT": "V", "SIGVOS": "V", "SIGVOS_PAIR": "V",
            "SIGBETA": "%", "SIGW": "%", "SIGL": "%",
            "SIGID": "%", "SIGID_PAIR": "%",
            "GAMMA": "", "SID": "A^2/Hz", "SVG": "V^2/Hz",
            "SFL": "A^2", "FKNEE": "Hz", "SVG_1HZ": "V^2/Hz"}

    def _mm_notes(self, snap):
        """
        What the numbers mean, said once and only where they bite.

        Three things a designer cannot read off the table itself: that
        mismatch ignores the finger count entirely, that part of it
        cannot be bought down with width, and which source dominates.
        """
        o = []
        fl = snap.get("_floor_pair")
        tot = snap.get("SIGID_PAIR")
        pL = snap.get("_dlnI_dlnL")
        o.append("")
        o.append(f"  mismatch is set by the DRAWN area "
                 f"{snap.get('_area', float('nan')):.3g} um2 and nothing "
                 f"else: the")
        o.append("  model divides by sqrt(W.L) with W the total width, "
                 "so the same")
        o.append("  device matches identically at 4 fingers or at 40.")
        if pL is not None and np.isfinite(pL):
            o.append(f"  the length term uses the table's own "
                     f"dlnID/dlnL = {pL:+.3f} at this bias,")
            o.append("  not the textbook -1: L moves the threshold as "
                     "well as the geometry.")
        if fl and tot and np.isfinite(fl) and np.isfinite(tot) and tot > 0:
            frac = fl / tot
            if frac > 0.2:
                o.append(f"  FLOOR: {fl:.3f} % of the pair's "
                         f"{tot:.3f} % is edge roughness, which is")
                o.append("  3 nm whatever the width. No width buys it "
                         "down; only a longer L does.")
        return o

    def _noise_notes(self, snap):
        """
        What the noise numbers mean and how they relate.

        The table stores sid (thermal PSD, white) and sfl (flicker
        coefficient, so S_flicker(f) = sfl/f). Both are output-referred
        drain current noise. Dividing by gm^2 gives the input-referred
        voltage noise that a differential pair or common-source stage
        presents. gamma = sid/(4kTgm) is the thermal excess noise factor:
        2/3 in long-channel theory, typically 0.6 to 1.0 in practice.
        """
        o = []
        g = snap.get("GAMMA")
        fk = snap.get("FKNEE")
        o.append("")
        if g is not None and np.isfinite(g):
            o.append(f"  gamma = {g:.3f} is the thermal noise excess factor "
                     f"(long-channel ideal: 2/3).")
        o.append("  SID is the thermal (white) drain noise; SFL is the "
                 "flicker coefficient.")
        o.append("  SVG = SID/gm^2 is the input-referred thermal floor "
                 "(V^2/Hz).")
        if fk is not None and np.isfinite(fk) and fk > 0:
            o.append(f"  below FKNEE = {fmt(fk, 'Hz')} flicker dominates; "
                     f"above it, thermal.")
        o.append("  SVG_1HZ = (SID + SFL)/gm^2 is the total input-referred "
                 "noise at 1 Hz.")
        o.append("  at any frequency f: SVG(f) = SVG + SFL/(gm^2 . f)")
        return o

    def __str__(self):
        o = ["=" * 68,
             f"  SIZING SOLUTION   {self['table']}",
             "=" * 68,
             "  asked for:"]
        for k, v in self["targets"].items():
            o.append(f"      {k:<10} = {fmt(v, self.UNIT.get(k.upper(), ''))}")
        o.append(f"  held fixed:  VDS = {self['VDS']:.4g} V"
                 f"   VSB = {self['VSB']:.4g} V")
        le = self.get("l_err")
        if le is not None and le[0] > 0.05:
            val, src = le
            o.append(f"  L = {fmt(self['L'],'m')} sits where the L axis "
                     f"is poorly resolved: the")
            o.append(f"  interpolation error of ID, gm and gds in this "
                     f"cell is up to {val:.3f} %,")
            o.append(f"  {src}.")
            o.append("  The cause is the model's reverse short-channel "
                     "hump, which is real")
            o.append("  physics and not smooth at 50 nm spacing, so a "
                     "finer grid there pays")
            o.append("  back at order 1.4 to 1.8 and not 4. On this "
                     "device the whole effect")
            o.append("  lives below about 0.85 um; every longer length "
                     "is at the float32")
            o.append("  floor. Pin L to a stored length if you need "
                     "the full accuracy.")
            o.append("")
        c = self.get("cond")
        if c is not None and np.isfinite(c):
            tag = ("well conditioned" if c < 30 else
                   "usable" if c < 300 else
                   "ILL CONDITIONED, treat with suspicion")
            o.append(f"  conditioning of the two bias constraints: "
                     f"{c:.1f}   {tag}")
            if c >= 300:
                o.append("    the two specs nearly carry the same "
                         "information, so L and VGS are only loosely")
                o.append("    pinned: a small change in either target "
                         "moves this device a lot. Replace one.")
        if self.get("n_roots", 1) > 1:
            o.append(f"  NOTE: {self['n_roots']} devices meet this "
                     f"specification; this is the one at the smallest L.")
        if self.get("vgs_roots", 1) > 1:
            # gm/ID peaks at the weak/moderate boundary and falls either
            # side of it, so a high target is met at two gate voltages.
            # The other one sits in deep weak inversion at a current
            # density millions of times lower, and would be built as a
            # device metres wide. Saying which was chosen costs a line.
            o.append(f"  NOTE: the bias constraint has "
                     f"{self['vgs_roots']} solutions in VGS. This is the "
                     f"one at the higher VGS,")
            o.append("        which is the one carrying current; the "
                     "other sits in deep weak inversion")
            o.append("        below the gm/ID peak and would need a far "
                     "wider device for the same ID.")
        o.append("")
        snap = self.get("snapped")
        o.append(f"  {'quantity':<10}{'asked for':>18}{'realised':>18}"
                 f"{'change':>12}")
        o.append("  " + "-" * 58)
        for k in self.REPORT:
            u = self.UNIT.get(k, "")
            a, b = self["ideal"].get(k), snap.get(k)
            if a and b and np.isfinite(a) and np.isfinite(b) and a != 0:
                d = f"{100*(b-a)/a:+.2f}%"
            else:
                d = ""
            o.append(f"  {k:<10}{fmt(a, u):>18}{fmt(b, u):>18}{d:>12}")
        # CGD arrives from the model as dQg/dVd, the off-diagonal term of
        # the capacitance matrix, and that term is negative by
        # construction: raising the drain pulls charge OFF the gate. The
        # MAGNITUDE is the Cgd a hand calculation means; the sign is
        # bookkeeping.
        #
        # It is deliberately NOT flipped. CGD is a leaf: CGG_TOT, CDTOT
        # and FT are built from cgg, cgsol, cgdol, cdb and cjd and never
        # from cgd, so flipping it would change no result -- but
        # check_verify.py compares this very number against ngspice's own
        # cgd, and fake_model.py reproduces the sign on purpose so the
        # self-tests see what the PDK produces. Flipping it in one place
        # and not the others is exactly how RAW_P['rg'] and the dW
        # literal both went wrong. One number, one place, one note.
        _cgd = snap.get("CGD")
        if _cgd is not None and np.isfinite(_cgd) and _cgd < 0:
            o.append("  CGD carries the model's own sign: it is dQg/dVd, "
                     "the off-diagonal")
            o.append(f"  capacitance, which is negative by construction. "
                     f"Read {fmt(abs(_cgd), 'F')} as the")
            o.append("  Cgd of a hand calculation. Nothing downstream uses "
                     "it: CGG_TOT, CDTOT")
            o.append("  and FT are built from cgg, the overlaps and the "
                     "junctions.")
        if self["ideal"].get("SIGID") is not None:
            o.append("  " + "-" * 58)
            for k in self.MM:
                u = self.UNIT.get(k, "")
                a, b = self["ideal"].get(k), snap.get(k)
                if a and b and np.isfinite(a) and np.isfinite(b) and a != 0:
                    d = f"{100*(b-a)/a:+.2f}%"
                else:
                    d = ""
                # a per cent does not take an engineering prefix: 0.34 %
                # printed as "342.5 m%" is a correct number nobody can
                # read. Volts still do.
                def sh(v):
                    if v is None or not np.isfinite(v):
                        return "--"
                    return f"{v:.4g} %" if u == "%" else fmt(v, u)
                o.append(f"  {k:<12}{sh(a):>16}{sh(b):>18}{d:>12}")
            o += self._mm_notes(snap)
        # ---- noise block: thermal + flicker, input-referred -----------
        if snap.get("GAMMA") is not None and np.isfinite(snap.get("GAMMA", float("nan"))):
            o.append("  " + "-" * 58)
            for k in self.NOISE:
                u = self.UNIT.get(k, "")
                b = snap.get(k)
                if b is None or not np.isfinite(b):
                    o.append(f"  {k:<12}{'--':>18}")
                    continue
                # gamma is dimensionless and small; show 3 decimals
                if k == "GAMMA":
                    o.append(f"  {k:<12}{b:>18.4f}")
                # squared units with SI prefixes are ambiguous:
                # aA^2/Hz could be (aA)^2 or a(A^2), so use scientific
                # notation for the spectral densities. FKNEE stays
                # in engineering form because Hz is not squared.
                elif u in ("A^2/Hz", "V^2/Hz", "A^2"):
                    o.append(f"  {k:<12}{b:>15.4e} {u}")
                else:
                    o.append(f"  {k:<12}{fmt(b, u):>18}")
            o += self._noise_notes(snap)
        o.append("")
        o.append(f"  realised as {self['ng']} fingers of "
                 f"{fmt(self['wf_drawn'], 'm')} drawn "
                 f"({fmt(self['wf_eff'], 'm')} effective each)")
        o.append(f"  the width wanted {self['ng_real']:.3f} fingers; "
                 f"rounding to {self['ng']} costs {self['ng_err']:+.2f} %")
        o.append("  on every quantity that scales with W")
        # Which of the three targets that rounding actually lands on.
        # A device can only have a whole number of fingers, so with three
        # targets and two free knobs one of them has to absorb it, and
        # naming it is the difference between a reported limit and a
        # mystery. The two bias constraints are met at the count as
        # BUILT, so the residual is all on this one.
        o.append(f"  and it lands on {self['width_knob']}: "
                 f"{' and '.join(self['bias_cons'])} "
                 f"{'are' if len(self['bias_cons']) > 1 else 'is'} met at "
                 f"the built")
        o.append(f"  finger count, so {self['width_knob']} is the target "
                 f"carrying the whole {abs(self['ng_err']):.2f} %. Ask for "
                 f"more")
        o.append("  current or a wider device if that is too coarse: the "
                 "step is one finger in")
        o.append(f"  {self['ng']}, so {50.0/max(self['ng'],1):.2f} % at "
                 f"worst.")
        if self.get("ng_sens"):
            tr = self.get("ng_trace") or []
            sn = self["ng_sens"]
            # two different reasons a target can depend on the count, and
            # saying the wrong one is worse than saying none
            why = ("because the drawn width books dW per finger"
                   if all(k.upper() in ("W", "AREA") for k in sn)
                   else "because the overlaps are affine in it and the "
                        "junctions follow its parity")
            o.append("")
            o.append(f"  {', '.join(sn)} depend"
                     f"{'' if len(sn) > 1 else 's'} on the "
                     f"finger count, not just on the bias,")
            o.append(f"  {why}. So the")
            o.append("  bias was re-solved at the count that came out, "
                     "until the count stopped")
            if tr:
                o.append("  moving:  "
                         + " -> ".join(f"{t:.3f}" for t in tr[:-1])
                         + f" -> {int(tr[-1])}")
            if not self.get("ng_settled", True):
                o.append("  the width sits halfway between two counts, so "
                         "it alternates between them;")
                o.append("  this is the one solved for, and the rounding "
                         "cost above is what it costs.")
        o.append("")
        o.append("  the finger WIDTH is fixed: it is the width this table")
        o.append("  was characterised at, and ID is proportional to the")
        o.append("  finger COUNT but not to the finger WIDTH.")
        # The two numbers used to be literals -- +16 % and 0.0000 % --
        # measured on sg13_hv_nmos at its 2 um finger, with the report
        # saying so. check_verify.py sections 4A and 4B measure both on
        # whatever table it is run against and leave them in
        # finger_laws.json, so they are read from there when they exist
        # and the attributed text stands when they do not. Nothing here
        # invents a number for a table that has not been checked.
        g = self.get("laws") or {}
        spr = g.get("id_per_ng_spread_pct")
        fwe = g.get("finger_width_effect_pct")
        ngs = g.get("ng_studied") or []
        if spr is not None and fwe is not None and ngs:
            floor = "" if g.get("wf_ref_inside_study") else " at least"
            o.append(f"  The size of that effect is{floor} {fwe:.3g} % on "
                     f"the current")
            # A spread needs its range: '22.8 %' says nothing until it
            # says over what change of finger width. The two ends come
            # from the study itself, so they cannot drift from it.
            wmax, wmin = g.get("wf_widest_studied"), g.get(
                "wf_narrowest_studied")
            if wmax and wmin:
                o.append(f"  density over finger widths "
                         f"{fmt(wmax, 'm')} down to {fmt(wmin, 'm')},")
                o.append(f"  against {spr:.4f} % spread over "
                         f"ng = {min(ngs)} to {max(ngs)} at a FIXED finger.")
            else:
                o.append(f"  density, against {spr:.4f} % spread over "
                         f"ng = {min(ngs)} to {max(ngs)},")
            o.append(f"  Both measured on THIS table against spice.")
            if not g.get("wf_ref_inside_study"):
                o.append("  'at least' because the study's narrowest "
                         "finger was wider")
                o.append(f"  than this table's own "
                         f"{fmt(g['wf_ref_eff'], 'm')} effective, and the "
                         f"effect")
                o.append("  grows as the finger narrows: a floor, not a "
                         "measurement.")
        else:
            o.append("  The size of that effect -- +16 % on the current")
            o.append("  density, and 0.0000 % spread over ng = 1 to 50 -- "
                     "was")
            o.append("  measured on sg13_hv_nmos at its 2 um finger. It has")
            o.append("  NOT been remeasured for this table. The law holds;")
            o.append("  the two numbers are the hv device's. Run "
                     "check_verify.py")
            o.append("  on this table and they become this table's.")
        if self.get("edge"):
            o.append("")
            o.append(f"  edge table : {self['edge']}")
            o.append("  adjacent fingers share their source and drain")
            o.append("  diffusions, so neither the overlaps nor the")
            o.append("  junctions are proportional to the count. The")
            o.append("  overlaps are affine in it; the junctions follow its")
            o.append("  PARITY, since region ng+1 is a source when ng is")
            o.append("  even and a drain when it is odd. Both laws are")
            o.append("  taken from the two tables, so Cgg, Cdtot and fT")
            o.append("  carry neither error. What ignoring them WOULD")
            # Same arrangement as the two numbers above: measured by
            # check_verify.py section 4D on this table, or attributed to
            # the hv device when it has not been run. cjd is quoted at an
            # EVEN count on purpose -- that is where the parity law bites,
            # and the odd counts hide it because an affine fit taken from
            # two odd counts reproduces every other odd count exactly.
            ci = (self.get("laws") or {}).get("cost_if_ignored_pct") or {}
            ce = ((self.get("laws") or {})
                  .get("cost_if_ignored_even_pct") or {})
            ex = (self.get("laws") or {}).get("law_exact_worst_pct")
            if ci.get("CGG_TOT") and ci.get("CDTOT") and ce.get("cjd"):
                o.append(f"  cost on THIS table, measured against spice:")
                o.append(f"  {ci['CGG_TOT']:.3g} % on Cgg, "
                         f"{ci['CDTOT']:.3g} % on Cdtot and, at an")
                o.append(f"  even count, {ce['cjd']:.3g} % on cjd. With the "
                         f"laws on, the")
                o.append(f"  worst error left is "
                         f"{(ex or 0.0):.4f} % across every count.")
            else:
                o.append("  cost -- 1.4 % on Cgg, 6.6 % on Cdtot and, at an")
                o.append("  even count, 44 % on cjd -- was measured on")
                o.append("  sg13_hv_nmos at its 2 um finger. A shorter "
                         "finger")
                o.append("  carries a LARGER end-term fraction, since z1 "
                         "and")
                o.append("  z2 do not shrink with it, so on a narrower "
                         "finger")
                o.append("  those three are floors and not estimates.")
                o.append("  Run check_verify.py on this table to replace "
                         "them.")
        else:
            o.append("")
            o.append("  NO edge table attached, so the overlap and junction")
            o.append("  capacitances are scaled as if proportional to the")
            o.append("  finger count. Measured against the model that")
            o.append("  leaves 1.4 % on Cgg and fT and 6.6 % on Cdtot.")
            o.append("  Build one with make_edge.py; it takes half an hour.")
        o.append("=" * 68)
        return "\n".join(o)


def edge_dir_for(lutdir):
    """
    The companion ng = 1 table, by convention: lut_nmos_hv_tt_27 pairs
    with edge_nmos_hv_tt_27. Found automatically rather than passed,
    because an optional argument for a correction this size is an
    argument that gets forgotten, and then Cdtot is quietly 6.6 % out.
    """
    base = os.path.basename(lutdir.rstrip("/"))
    stem = base[4:] if base.startswith("lut_") else base
    cand = os.path.join(os.path.dirname(lutdir.rstrip("/")),
                        "edge_" + stem)
    return cand if os.path.isdir(cand) else None


class Sizer:
    def __init__(self, lutdir, edge=True):
        self.lut = LUT(lutdir)
        self.mm = Mismatch(lutdir)
        self.dir = lutdir
        self.edge_dir = None
        if edge:
            cand = edge if isinstance(edge, str) else edge_dir_for(lutdir)
            if cand:
                self.lut.attach_edge(cand)
                self.edge_dir = cand
        # The finger count the constraints are evaluated at. None means
        # the table's own reference, which is the only correct answer
        # for a quantity that really is width-independent, and is also
        # the fastest path. solve() sets it when a specification turns
        # out to depend on the count; see the long note there.
        self._ng_eval = None
        # The per-length interpolation error, if l_axis_probe.py has been
        # run on this table. Optional on purpose: a table without it
        # still sizes, it just cannot warn.
        self.l_err = None
        self.l_src = None
        self.l_stale = False
        try:
            with open(os.path.join(lutdir, "l_accuracy.json")) as fh:
                d = json.load(fh)
            if list(d["L"]) == list(self.lut.L):
                est = d.get("err_pct") or []
                mea = d.get("err_meas") or []
                nme = d.get("n_meas") or []
                # A measurement is only valid for the interpolation
                # path it was taken on. check_probe.py stamps that path
                # into measured_scheme; if it is missing (an old merge)
                # or disagrees with what the engine does NOW, the
                # numbers describe a path nobody takes -- which is how
                # an 80 % figure came to sit on a table accurate to a
                # fraction of a per cent. Fall back to the bound then.
                sch = d.get("measured_scheme")
                now = dict(cubic=bool(LUT.CUBIC),
                           slope=bool(LUT.SLOPE))
                self.l_stale = bool(mea) and sch != now
                if self.l_stale:
                    mea, nme = [], []
                # MEASURED DISPLACES ESTIMATED. l_axis_probe.py writes a
                # table-only bound that is deliberately conservative
                # where its order test fails -- on the real table it read
                # 0.615 % in a cell SPICE measured at 0.014 %, 44x loose.
                # check_probe.py merges the SPICE numbers into the same
                # file as err_meas, so prefer those, and keep the bound
                # only for lengths the probe never visited. A cell
                # measured on fewer than 8 points is not a measurement:
                # a p90 of 3 samples is noise, so the bound stands.
                self.l_err, self.l_src = [], []
                for k in range(len(self.lut.L)):
                    e = est[k] if k < len(est) else None
                    m = mea[k] if k < len(mea) else None
                    n = nme[k] if k < len(nme) else 0
                    if m is not None and n >= 8:
                        self.l_err.append(float(m))
                        self.l_src.append(f"measured against SPICE on "
                                          f"{int(n)} points")
                    else:
                        self.l_err.append(None if e is None else float(e))
                        self.l_src.append(
                            "a table-only upper bound; the measurement "
                            "on file was taken on another interpolation "
                            "path, re-run check_probe.py"
                            if self.l_stale else
                            "a table-only upper bound; "
                            "run check_probe.py to measure it")
        except Exception:                                    # noqa: BLE001
            self.l_err = self.l_src = None
        # The geometry laws as check_verify.py measured them ON THIS
        # TABLE, by the same arrangement as l_accuracy.json above: a
        # table without the file still sizes, it just has to say that the
        # numbers in its report belong to another device. Optional on
        # purpose, and never faked -- an absent file prints the old,
        # attributed text and not an invented figure.
        self.laws = None
        try:
            with open(os.path.join(lutdir, "finger_laws.json")) as fh:
                g = json.load(fh)
            # A file left behind by a different table is worse than none:
            # it would be quoted with this table's name on it.
            if g.get("device") == self.lut.device:
                self.laws = g
        except Exception:                                    # noqa: BLE001
            self.laws = None
        # False while the solver is still looking for a finger count, so
        # the junction law is read on its continuous branch instead of
        # stepping between even and odd. True for anything that is a
        # real device. See gmid_lut.finger_regions.
        self._parity = True

    # -- the two primitives ------------------------------------------------
    def _val(self, knob, L, VGS, VDS, VSB):
        """
        Knob value in REFERENCE-WIDTH units, so that the width-independent
        combinations built out of these are the real device's.

        With _ng_eval set, the value is read for a device of that many
        whole reference fingers and the pure power-law part of the width
        scaling is then divided back out. For every quantity that really
        is a power law in W this returns exactly what the plain reference
        read returns -- the scale factor cancels to the last bit, because
        ng whole fingers give scale = ng/ng_ref exactly. For the ones the
        edge correction made affine it does not cancel, and what comes
        back carries the finger-count dependence that is actually there.

        That is the point: the quantity the constraint is built from stops
        being a function of (L, VGS) alone and becomes a function of
        (L, VGS, ng), so solve() has to feed the realised count back in.
        """
        k = knob.upper()
        if k == "L":
            return np.asarray(L, float)
        if k == "VGS":
            return np.asarray(VGS, float)
        ng = self._ng_eval
        if ng is None:
            return self.lut.eval(k, L=L, VGS=VGS, VDS=VDS, VSB=VSB)
        v = self.lut.eval(k, L=L, VGS=VGS, VDS=VDS, VSB=VSB,
                          W=ng * (self.lut.W_nom / self.lut.ng), ng=ng,
                          parity=self._parity)
        p = power_of(k)
        if p == 0:
            return v
        return v / (ng / self.lut.ng) ** p

    def l_accuracy_at(self, L):
        """
        The measured interpolation error, in per cent, of a solution at
        this length. None when it was never measured.

        L sits inside a stored cell, and either of that cell's two nodes
        can be the one whose neighbourhood is badly resolved, so the
        worse of the two is what the solution carries.
        """
        if not self.l_err:
            return None
        g = self.lut.L
        i = int(np.clip(np.searchsorted(g, L) - 1, 0, len(g) - 2))
        cand = [(self.l_err[j], (self.l_src or [None] * len(g))[j])
                for j in (i, i + 1) if self.l_err[j] is not None]
        return max(cand) if cand else None

    def _ng_sensitive(self, knob):
        if knob.upper() in MM_KNOBS:
            # Sensitive, for the same reason W and AREA are. The solve
            # itself reads only gm/ID and L, which are both
            # finger-independent -- but it answers in a DRAWN width,
            # and this method's caller works in effective ones. The
            # conversion W_eff = W_drawn - ng.dW needs the count, and
            # using the reference count on the first pass left 1.1 % on
            # a sigma target whose own ideal column should have been
            # exact. Half of 13 dW in 35 um, under a square root.
            return True
        return self._ng_sensitive_inner(knob)

    def _ng_sensitive_inner(self, knob):
        """
        Does this specification's value depend on the finger count?

        Two separate reasons it can:

          * the edge correction. fT, Cgg_tot and Cdtot are built from
            capacitances that are affine in the count, so they move with
            it. gmid_lut answers this from the DERIVED lambdas themselves.

          * the drawn-to-effective width conversion, which is where W as
            a specification comes in. Weff = W_drawn - ng*dW, and dW is
            0.06 um against a 1.94 um finger, so assuming 5 fingers while
            building 30 books 0.3 um of narrowing instead of 1.8 um. On a
            60 um device that is 2.5 % of the width, in the direction
            that makes it too strong. This one has nothing to do with the
            edge table and was there all along.
        """
        k = str(knob).upper()
        if k in ("W", "AREA"):
            return True
        if k in GEOM or k in ("VDS", "VSB", "WEFF"):
            return False
        return self.lut.affine_sensitive(k)

    def _vgs_hi(self, VSB):
        """largest VGS the stored grid covers at this VSB"""
        return float(self.lut.meta["vmax"]) - float(VSB) - 1e-9

    # -- step 1: VGS from one bias constraint, VECTORISED over L ------------
    def _vgs_curve(self, con, Ls, VDS, VSB, n_scan=129, iters=24):
        """
        Solve con(L, VGS) == 1 for VGS at EVERY L in Ls at once.

        Vectorised on purpose. A scalar bisection per L costs 37 x 34
        separate table reads per solve, which turned out to be about a
        second; doing all the L values in lockstep makes it 34 reads of
        width 37, about 40 ms. Same arithmetic, same answer.

        The whole valid VGS range is scanned for a sign change rather
        than assuming monotonicity. Most width-independent quantities are
        monotonic in VGS, but 'most' is not 'all', and assuming it would
        return a confident wrong device. NaN columns -- an L where the
        constraint is unreachable, or outside the stored grid -- come back
        as NaN, never as a clipped edge value.
        """
        Ls = np.atleast_1d(np.asarray(Ls, float))
        nL = Ls.size
        g = np.linspace(0.0, self._vgs_hi(VSB), n_scan)
        LL = np.broadcast_to(Ls, (n_scan, nL))
        GG = np.broadcast_to(g[:, None], (n_scan, nL))
        f = np.asarray(con(LL, GG), float) - 1.0

        fin = np.isfinite(f)
        # An EXACT root is a root. Looking only for a strict sign change
        # misses it, because at the root the residual is 0 and 0*x is not
        # < 0. That is not a corner case: a designer types round numbers,
        # and round numbers land on grid nodes.
        hit = fin & (np.abs(f) <= 1e-12)
        exact = hit.any(axis=0)

        sg = np.where(fin, np.sign(f), 0.0)
        ch = fin[:-1] & fin[1:] & (sg[:-1] * sg[1:] < 0)
        has = ch.any(axis=0)

        # WHICH ROOT, when there is more than one.
        #
        # gm/ID is not monotone in VGS. It is 1/(n.Ut), and n = 1 +
        # Cdep/Cox falls as the depletion region widens, so gm/ID RISES
        # through weak inversion, peaks at the weak/moderate boundary,
        # and only then falls. Measured on this PDK at VDS = 0.9, VSB =
        # 0: the 0.45 um device peaks at 28.12 (VGS = 0.45), the 0.8 um
        # at 29.15, the 1.3 um at 28.73. Every target above the shoulder
        # therefore has TWO roots, and asking for gm/ID = 25 at 0.45 um
        # offers VGS = 0.0209 or VGS = 0.5757.
        #
        # Taking the first, which this did, returned the 0.0209 one: a
        # device carrying 4e-8 A/m against the other's 0.21 A/m, five
        # million times less current for the same requested gm/ID. For
        # ID = 100 uA that branch asks for W = 2500 METRES.
        #
        # The rule is not "take the last one". ID is strictly monotone
        # increasing in VGS for any MOSFET at fixed VDS and VSB, so
        # among the roots of ANY constraint the highest VGS is the one
        # carrying the most current -- the most strongly-on device that
        # satisfies the request. That is a statement about the device,
        # true for every quantity this solver is handed and not only for
        # gm/ID; it merely happens to land on the last root because of
        # that monotonicity.
        #
        # A target above the peak still has NO root and still comes back
        # NaN. The curve ends where the device ends, which is the point.
        last = (n_scan - 2) - np.argmax(ch[::-1], axis=0)
        gexact = g[n_scan - 1 - np.argmax(hit[::-1], axis=0)]
        A = np.where(has, g[last], 0.0)
        B = np.where(has, g[np.minimum(last + 1, n_scan - 1)], 1.0)

        # how many roots each column had, so a caller can say so
        self._roots = ch.sum(axis=0) + hit.sum(axis=0)
        self._multi = int((self._roots > 1).sum())

        FA = np.asarray(con(Ls, A), float) - 1.0
        for _ in range(iters):
            M = 0.5 * (A + B)
            FM = np.asarray(con(Ls, M), float) - 1.0
            left = np.isfinite(FM) & ((FA * FM) <= 0)
            B = np.where(left, M, B)
            A = np.where(left, A, M)
            FA = np.where(left, FA, FM)
        out = np.where(has, 0.5 * (A + B), np.nan)
        # an exact hit on a node and a bracketed root can both exist in
        # the same column; the same rule picks between them
        out = np.where(exact & ~has, gexact, out)
        return np.where(exact & has, np.maximum(gexact, out), out)

    # -- step 2: L from the second constraint along that curve -------------
    _n_roots = 1          # roots of the SECOND constraint, along L
    _roots = None         # roots of the FIRST constraint, per L column
    _multi = 0

    def _solve_L(self, con1, con2, VDS, VSB, rounds=4, per=33):
        """
        con2 evaluated along the con1 curve is a function of L alone.
        Bracket its sign change on the stored L grid, then refine by
        re-gridding inside the bracket, geometrically because the L axis
        is geometric. Four rounds of 33 points is a factor 32^3 = 3e-5 in
        relative L, well inside the table's own 0.028 % on ID.
        """
        Ls = self.lut.L.copy()
        lo = hi = None
        for rd in range(rounds):
            vg = self._vgs_curve(con1, Ls, VDS, VSB)
            with np.errstate(invalid="ignore"):
                res = np.asarray(con2(Ls, np.where(np.isfinite(vg), vg, 0.0)),
                                 float) - 1.0
            res = np.where(np.isfinite(vg), res, np.nan)
            fin = np.isfinite(res)
            if fin.sum() < 2:
                if rd == 0:
                    raise NoSolution(
                        "the first constraint has no solution in VGS at any "
                        "stored L for this VDS and VSB.")
                break
            hit = fin & (np.abs(res) <= 1e-12)
            if hit.any():
                j = int(np.argmax(hit))
                lo = hi = float(Ls[j])
                if rd == 0:
                    self._n_roots = int(hit.sum())
                break
            s = np.where(fin, np.sign(res), 0.0)
            ch = fin[:-1] & fin[1:] & (s[:-1] * s[1:] < 0)
            if not ch.any():
                if rd > 0:
                    break
                rel = res[fin] + 1.0
                j = int(np.nanargmin(np.abs(res)))
                raise NoSolution(
                    "the second constraint cannot be met at any L.\n"
                    f"  along the first constraint it only reaches "
                    f"{rel.min():.4g} .. {rel.max():.4g} times your target\n"
                    f"  (closest at L = {fmt(Ls[j], 'm')})")
            if rd == 0:
                # more than one sign change means more than one device
                # meets the specification. Taking the first silently
                # would hide the other one.
                self._n_roots = int(ch.sum())
            i = int(np.argmax(ch))
            lo, hi = float(Ls[i]), float(Ls[i + 1])
            Ls = np.geomspace(lo, hi, per)
        if lo is None:
            raise NoSolution("the contour broke down while refining L.")
        Lsol = lo if lo == hi else math.exp(0.5 * (math.log(lo)
                                                   + math.log(hi)))
        vgs = float(self._vgs_curve(con1, np.array([Lsol]), VDS, VSB)[0])
        self._vgs_roots = int(self._roots[0]) if self._roots is not None \
            else 1
        if not np.isfinite(vgs):
            raise NoSolution("the contour broke down while refining L.")
        return Lsol, vgs

    # -- one complete solve at an ASSUMED finger count ---------------------
    def _one_pass(self, cons, wname, wp, wtarget, VDS, VSB, ng_eval,
                  parity=True):
        """
        The two bias constraints, then the width, with the finger count
        taken to be ng_eval. Returns (L, VGS, w_eff) where w_eff is the
        effective width the width-setting constraint asks for at that
        bias.

        ng_eval may be fractional. Nothing in here needs a whole number,
        and the outer loop deliberately converges on the continuous value
        before rounding once, at the end: iterating on the ROUNDED count
        turns a smooth fixed point into a step function, which can sit
        and oscillate between two neighbouring integers forever.
        """
        self._ng_eval = ng_eval
        self._parity = parity
        Lsol, vgs = self._solve_L(cons[0], cons[1], VDS, VSB)
        ngw = self.lut.ng if ng_eval is None else ng_eval
        if wname.upper() in ("W", "WEFF"):
            # Weff = W_drawn - ng*dW, so this needs the count too.
            w_eff = wtarget if wname.upper() == "WEFF" \
                else self.lut.weff(wtarget, ngw)
        elif wname.upper() in MM_KNOBS:
            # Mismatch sets the DRAWN area, so this branch is the only
            # one that solves for a drawn width and converts down,
            # rather than scaling an effective one. And it is the only
            # knob with a floor: a target at or below it is met by no
            # width at all, so saying 'no, and here is the length that
            # would work' is part of answering.
            gmid = float(self._val("GM/ID", np.array([Lsol]),
                                   np.array([vgs]), VDS, VSB)[0])
            if not np.isfinite(gmid) or gmid <= 0:
                raise NoSolution("gm/ID is not positive at the solved "
                                 "bias, so mismatch cannot set the width.")
            pL = self.mm.dlnI_dlnL(self.lut, Lsol, vgs, VDS, VSB)
            if not np.isfinite(pL):
                raise NoSolution(
                    "dlnID/dlnL could not be taken at this length, so "
                    "the mismatch length term is unknown. Pin L to a "
                    "stored value.")
            w_drawn = self.mm.W_for(wtarget, wname.upper(), Lsol, gmid, pL)
            if w_drawn is None:
                fl = self.mm.floor_of(wname.upper(), Lsol, gmid, pL)
                lb = self.mm.L_for_floor(self.lut, wtarget, vgs, VDS, VSB) \
                    if wname.upper().endswith("_PAIR") else None
                u = "%" if wname.upper().startswith("SIGID") else "V"
                extra = ("" if lb is None else
                         f"\n  The shortest stored length whose floor is "
                         f"below it is {lb*1e6:g} um.")
                raise NoSolution(
                    f"{wname} = {wtarget:g} {u} is at or below the floor "
                    f"of {fl:.4g} {u} at L = {Lsol*1e6:g} um.\n"
                    f"  The floor is edge roughness: sigma(dL) is "
                    f"{self.mm.d['S_DL']*1e9:g} nm whatever the width, so "
                    f"no width reaches this.\n"
                    f"  Only a longer L does.{extra}")
            w_eff = self.lut.weff(w_drawn, ngw)
        else:
            vw = float(self._val(wname, np.array([Lsol]), np.array([vgs]),
                                 VDS, VSB)[0])
            if not np.isfinite(vw) or vw == 0:
                raise NoSolution(f"{wname} is {vw} at the solved bias, so W "
                                 f"cannot be scaled from it.")
            w_eff = self.lut.W_ref_eff * (wtarget / vw) ** (1.0 / wp)
        return Lsol, vgs, w_eff

    # -- conditioning ------------------------------------------------------
    def _jac(self, cons, L, VGS, VDS, VSB, du=2e-3, dv=1e-3):
        """
        2x2 Jacobian of the two normalised residuals with respect to
        (ln L, VGS), by central differences. Both residuals are already
        divided by their target, so the entries are dimensionless
        relative sensitivities and the condition number means something.
        """
        J = np.empty((2, 2))
        Lp, Lm = L * math.exp(du), L * math.exp(-du)
        for r, con in enumerate(cons):
            a = float(np.asarray(con(np.array([Lp]), np.array([VGS]))).ravel()[0])
            b = float(np.asarray(con(np.array([Lm]), np.array([VGS]))).ravel()[0])
            J[r, 0] = (a - b) / (2 * du)
            a = float(np.asarray(con(np.array([L]), np.array([VGS + dv]))).ravel()[0])
            b = float(np.asarray(con(np.array([L]), np.array([VGS - dv]))).ravel()[0])
            J[r, 1] = (a - b) / (2 * dv)
        return J

    def cond_at(self, cons, L, VGS, VDS, VSB):
        J = self._jac(cons, L, VGS, VDS, VSB)
        if not np.all(np.isfinite(J)):
            return None
        sv = np.linalg.svd(J, compute_uv=False)
        if sv[-1] <= 0:
            return np.inf
        return float(sv[0] / sv[-1])

    def _pre_cond(self, cons, VDS, VSB, n=5):
        """
        Median conditioning over a few interior points, used to reject a
        dependent PAIR before any solving. Median, not minimum: one
        badly conditioned spot is normal, a badly conditioned grid is a
        dependent pair.
        """
        Lg = self.lut.L
        hi = self._vgs_hi(VSB)
        out = []
        for L in np.geomspace(Lg[2], Lg[-3], n):
            for v in (0.35 * hi, 0.6 * hi):
                c = self.cond_at(cons, float(L), float(v), VDS, VSB)
                if c is not None and np.isfinite(c):
                    out.append(c)
        return float(np.median(out)) if out else None

    # -- the public entry point -------------------------------------------
    def solve(self, VDS, VSB=0.0, ng=None, snap=True, **knobs):
        # ng is NOT an input any more: it is the answer. The finger width
        # belongs to the table, and the number of fingers is what sets W.
        if ng is not None and int(ng) != int(self.lut.ng):
            raise ValueError(
                f"ng is an output of this solver, not an input. The table "
                f"was characterised with {self.lut.ng} fingers of "
                f"{fmt(self.lut.W_nom / self.lut.ng, 'm')} drawn, and the "
                f"current density changes by up to 16 % if the finger "
                f"width changes, so the finger width is fixed and the "
                f"COUNT is solved for. To design at a different finger "
                f"width, re-characterise the table at that width.")
        if len(knobs) != 3:
            raise ValueError(
                f"exactly 3 specifications are needed, got {len(knobs)}: "
                f"{list(knobs)}. VDS and VSB are separate, and the five "
                f"degrees of freedom are VGS, VDS, VSB, W, L.")
        names = list(knobs)
        pw = [power_of(n) for n in names]

        if all(p == 0 for p in pw):
            raise ValueError(
                "all three specifications are width-independent, so they "
                "put 3 constraints on 2 unknowns (VGS and L) and leave W "
                "unconstrained. Replace one with a quantity that scales "
                "with W: ID, gm, gds, a capacitance, ro, or W itself.")

        # the width setter: prefer |p| == 1, and prefer ID when present
        order = sorted(range(3), key=lambda i: (pw[i] == 0, abs(pw[i]),
                                                names[i].upper() != "ID"))
        iw = order[0]
        mmi = [i for i, n in enumerate(names) if n.upper() in MM_KNOBS]
        if mmi:
            if not self.mm:
                raise ValueError("  " + self.mm.why_not)
            if len(mmi) > 1:
                raise ValueError(
                    "two mismatch specifications constrain the same "
                    "area, so they are the same constraint asked twice. "
                    "Give one.")
            # It MUST set the width. The other two are then turned into
            # bias constraints by dividing out a power of the width
            # setter's value -- which only works if the width setter is
            # a power law in W. Mismatch is not one, so any OTHER
            # width-dependent knob alongside it cannot be reduced that
            # way, and the honest answer is to say so rather than to
            # produce a number by a route that does not apply.
            iw = mmi[0]
            bad = [names[i] for i in range(3)
                   if i != iw and pw[i] != 0]
            if bad:
                raise ValueError(
                    f"{names[iw]} and {', '.join(bad)} both depend on the "
                    f"width, and mismatch is not a power law in it, so "
                    f"the two cannot be combined by the elimination this "
                    f"solver uses.\n  Specify {names[iw]} with two "
                    f"width-INDEPENDENT quantities instead, such as "
                    f"GM/ID and L.\n  (Asking for {names[iw]} together "
                    f"with {bad[0]} and letting L float is a solvable "
                    f"problem, but by a different route than this one, "
                    f"and it is not implemented.)")
        wname, wp = names[iw], pw[iw]
        others = [i for i in range(3) if i != iw]

        # bias constraint i:  K_i * K_w^(-p_i/p_w)   is width-independent
        cons, cons_txt, cons_t = [], [], []
        for i in others:
            e = -pw[i] / wp
            ni, nw = names[i], wname

            def mk(ni=ni, nw=nw, e=e):
                def f(L, VGS):
                    a = self._val(ni, L, VGS, VDS, VSB)
                    if e == 0:
                        return a
                    b = self._val(nw, L, VGS, VDS, VSB)
                    return a * np.power(b, e)
                return f
            t = knobs[ni] * (knobs[nw] ** e if e else 1.0)
            if t == 0:
                raise ValueError(f"target for {ni} cannot be zero")
            cons.append(lambda L, VGS, f=mk(), t=t: f(L, VGS) / t)
            cons_txt.append(ni if e == 0 else f"{ni}*{nw}^{e:g}")
            cons_t.append(t)

        # Two constraints that are not independent pin only one of VGS
        # and L, and the honest test for that is the CONDITIONING of the
        # pair, not a special case per quantity. The first version here
        # compared the two residuals at two probe points and looked for
        # proportionality, which is the wrong shape: if con2 = C/con1
        # then the ratios are reciprocal, not equal, so gm/ID together
        # with V* = 2/(gm/ID) sailed straight through it.
        #
        # The Jacobian of (f1, f2) with respect to (ln L, VGS) is
        # singular exactly when the two constraints carry the same
        # information, and merely ill-conditioned when they nearly do --
        # which is the real hazard, because gm/ID and Vdsat are nearly
        # the same constraint and that pair was landing on the wrong
        # contour intersection with an 11 % error in L. Both cases fall
        # out of one number.
        cnd = self._pre_cond(cons, VDS, VSB)
        if cnd is not None and cnd > 1e7:
            raise ValueError(
                f"{cons_txt[0]} and {cons_txt[1]} are not independent "
                f"constraints: across the stored grid their Jacobian has "
                f"condition number {cnd:.2e}, so together they pin only "
                f"one of VGS and L and nothing fixes the other. gm/ID and "
                f"V* are an example, since V* = 2/(gm/ID).")

        # ------------------------------------------------------------------
        # DOES THE ANSWER FEED BACK INTO THE QUESTION?
        #
        # The whole structure of this solver -- two bias constraints on
        # (VGS, L), then the width in closed form -- rests on the bias
        # constraints not knowing about the width. That was true while
        # every parameter was a clean power law in W. The edge correction
        # made it false for a few of them: adjacent fingers share their
        # diffusions, so the overlap and junction capacitances are AFFINE
        # in the finger count, and anything built on them moves with the
        # count. fT is the one that matters, being
        # gm/(2*pi*(cgg+cgsol+cgdol)) -- a numerator proportional to the
        # count over a denominator affine in it, drifting about 1.4 %
        # across the useful range.
        #
        # So an fT specification solved at the table's 5-finger reference
        # and then realised at 34 fingers misses by that drift, and it
        # misses INVISIBLY: every internal consistency check still passes,
        # because the sizer and the table agree with each other. It only
        # shows up against the specification, which is the one comparison
        # that is not self-referential.
        #
        # The cure is to put the count back into the constraints and
        # iterate. It is a contraction, and a strong one: the map is
        # 'count -> count' with a derivative of order the drift itself,
        # a few per cent, so it settles in two or three passes.
        #
        # Specifications that do not depend on the count skip all of this
        # and take exactly the path they took before, at exactly the old
        # cost. Which ones those are is asked, not assumed: gmid_lut reads
        # the dependency set out of the DERIVED lambdas, so a quantity
        # added later is classified without anyone remembering to.
        # ------------------------------------------------------------------
        sens = [n for n in names if self._ng_sensitive(n)]
        wf_eff = self.lut.W_ref_eff / self.lut.ng      # 1.94 um
        wf_drawn = self.lut.W_nom / self.lut.ng        # 2 um
        wtarget = knobs[wname]
        trace, settled = [], True

        try:
            if not sens:
                Lsol, vgs, w_eff = self._one_pass(
                    cons, wname, wp, wtarget, VDS, VSB, None)
                ng_real = w_eff / wf_eff
                ng_use = max(1, int(round(ng_real)))
            else:
                ng_eval = float(self.lut.ng)
                trace.append(ng_eval)
                for _ in range(12):
                    # parity OFF here: see finger_regions. The junctions
                    # step between an even and an odd law, and chasing a
                    # fixed point across that step oscillates forever.
                    Lsol, vgs, w_eff = self._one_pass(
                        cons, wname, wp, wtarget, VDS, VSB, ng_eval,
                        parity=False)
                    nxt = max(1.0, w_eff / wf_eff)
                    trace.append(nxt)
                    # 0.1 % in the COUNT, not in the answer. What the
                    # count feeds is the end term, which is a few per cent
                    # of the total, so a relative error d in the count
                    # moves the answer by at most 0.07*d: 1e-3 here is
                    # 7e-5 on fT, which is a factor four below the
                    # table's own 0.03 % on ID. Chasing it to 1e-7 bought
                    # nothing and cost a whole extra pass.
                    done = abs(nxt - ng_eval) <= 1e-3 * nxt
                    ng_eval = nxt
                    if done:
                        break
                # Round ONCE, here, and then solve the bias one more time
                # at the whole number of fingers the device will really
                # have. That puts the two bias constraints on the device
                # AS BUILT, and leaves the entire rounding error on the
                # one width-setting quantity -- where the report already
                # prints it, instead of smeared silently over all three.
                ng_use = max(1, int(round(ng_eval)))
                tried = {}
                for _ in range(5):
                    # parity ON from here: these are real devices.
                    Lsol, vgs, w_eff = self._one_pass(
                        cons, wname, wp, wtarget, VDS, VSB,
                        float(ng_use), parity=True)
                    ng_real = w_eff / wf_eff
                    tried[ng_use] = (abs(ng_real - ng_use) / ng_real,
                                     Lsol, vgs, w_eff, ng_real)
                    nn = max(1, int(round(ng_real)))
                    if nn == ng_use:
                        break
                    if nn in tried:
                        # Each of two counts points at the other. With a
                        # junction quantity setting the width that is not
                        # a failure to converge but an honest tie: the
                        # law steps with the PARITY of the count, so the
                        # even and the odd solution sit about 0.75 of a
                        # finger apart and neither rounds to itself.
                        # Keep whichever lands closer to the width it
                        # asked for, and say so.
                        settled = False
                        best = min(tried, key=lambda n: tried[n][0])
                        _, Lsol, vgs, w_eff, ng_real = tried[best]
                        ng_use = best
                        break
                    ng_use = nn
                else:
                    # ran out of passes: check the same thing the two
                    # breaks check, so the flag cannot be left true on a
                    # count that does not round to itself.
                    if max(1, int(round(ng_real))) != ng_use:
                        settled = False
                trace.append(float(ng_use))
            w_eff_used = ng_use * wf_eff
            w_drawn_used = ng_use * wf_drawn
            # The 'asked for' column: the device the specification wants,
            # which needs a FRACTIONAL finger count -- 62.745 fingers.
            # dW used to be booked at the ROUNDED count here, which left
            # the drawn width and the count disagreeing by half a
            # finger's worth. The edge correction rightly refuses that
            # (it has no defined end term for a width that is not a whole
            # number of its own fingers), report_at swallowed the
            # exception, and the entire 'asked for' column came out as a
            # page of dashes -- the one column a designer reads. The
            # affine law c*ng + e extends to a fractional count without
            # any trouble; it is the LAYOUT that cannot, and that is what
            # the rounding line is for.
            w_drawn_cont = w_eff + ng_real * self.lut.dW
            cond = self.cond_at(cons, Lsol, vgs, VDS, VSB)
        finally:
            self._ng_eval = None
            self._parity = True

        def report_at(w, ngg):
            d = {}
            for k in Result.REPORT + Result.NOISE:
                try:
                    v = self.lut.eval(k, L=np.array([Lsol]),
                                      VGS=np.array([vgs]), VDS=VDS,
                                      VSB=VSB, W=w, ng=ngg)
                    d[k] = float(np.asarray(v).ravel()[0])
                except Exception:
                    d[k] = None
            d["L"], d["W"] = Lsol, w
            # mismatch: computed from the DRAWN area and gm/ID, never
            # looked up, so it is added after the table loop and not
            # inside it
            g = d.get("GM/ID")
            if self.mm and g and np.isfinite(g) and g > 0:
                d.update(self.mm.at(self.lut, Lsol, vgs, VDS, VSB, w, g))
            return d

        out = Result(table=self.dir, targets=dict(knobs), VDS=VDS, VSB=VSB,
                     ng=ng_use, ng_real=ng_real, L=Lsol, VGS=vgs,
                     ng_sens=sens, ng_trace=trace, ng_settled=settled,
                     W=w_drawn_used, W_eff=w_eff_used,
                     W_cont=w_drawn_cont, W_eff_cont=w_eff,
                     wf_eff=wf_eff, wf_drawn=wf_drawn,
                     ng_ref=self.lut.ng, edge=self.edge_dir,
                     width_knob=wname, bias_cons=cons_txt, cond=cond,
                     n_roots=self._n_roots,
                     vgs_roots=getattr(self, "_vgs_roots", 1),
                     l_err=self.l_accuracy_at(Lsol),
                     laws=self.laws,
                     ideal=report_at(w_drawn_cont, ng_real),
                     snapped=report_at(w_drawn_used, ng_use))
        # the quantisation cost, stated rather than buried
        q = 100.0 * (ng_use - ng_real) / ng_real
        out["ng_err"] = q
        if ng_use <= 4:
            out["warning"] = (
                f"only {ng_use} finger{'s' if ng_use != 1 else ''}, so the "
                f"width is quantised in steps of {100.0/ng_use:.0f} %. "
                f"Ask for more current, a shorter L or a lower gm/ID if "
                f"that is too coarse.")
        elif ng_use > 200:
            out["warning"] = (
                f"{ng_use} fingers is a large device; consider several in "
                f"parallel, which is the same thing electrically.")
        return out


# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("example:")
        print("  python3 gmid_size.py lut_nmos_hv_tt_27 VDS=0.9 VSB=0 "
              "GM/ID=15 FT=10G ID=50u")
        return 1
    lutdir, args = argv[1], argv[2:]
    fixed, knobs = {"VSB": 0.0}, {}
    ng = None
    for a in args:
        if "=" not in a:
            sys.exit(f"expected name=value, got {a!r}")
        k, v = a.split("=", 1)
        if k.upper() in ("VDS", "VSB"):
            fixed[k.upper()] = parse_value(v)
        elif k.lower() == "ng":
            ng = int(parse_value(v))
        else:
            knobs[k] = parse_value(v)
    if "VDS" not in fixed:
        sys.exit("VDS is required: it is part of the bias plan, not a spec")
    s = Sizer(lutdir)
    try:
        r = s.solve(VDS=fixed["VDS"], VSB=fixed["VSB"], ng=ng, **knobs)
    except (ValueError, NoSolution) as e:
        print("=" * 68)
        print("  NO SOLUTION")
        print("=" * 68)
        print(f"  {e}")
        print("=" * 68)
        return 2
    print(r)
    if r.get("warning"):
        print(f"  note: {r['warning']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
