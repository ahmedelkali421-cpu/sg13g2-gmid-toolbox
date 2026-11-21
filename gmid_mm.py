#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | MISMATCH, FROM THE STORED TABLE
=============================================================================
No simulation. The coefficients come from mismatch.json, which
make_mismatch.py extracts from the PDK and refuses to write unless the
formula there is the one assumed here.

WHAT PELGROM GIVES, AND WHAT IT DOES NOT

    sigma(dVt)      = A_VT   / sqrt(W . L)      W, L in um, DRAWN
    sigma(dbeta/b)  = A_BETA / sqrt(W . L)
    sigma(dW)       = S_DW                      absolute, metres
    sigma(dL)       = S_DL                      absolute, metres

The first two shrink with area. The last two do NOT: they are edge
roughness, the same 3 nm on a 2 um finger as on a 200 um one. So there is
a mismatch FLOOR that no width can buy down, set by L alone, and that is
a design fact the tool should say out loud rather than leave the designer
to discover.

The finger count does not appear. The model divides by sqrt(m.l.w), with
w the TOTAL width, so a device of a given W and L matches the same
whether it is drawn as 4 fingers or 40. Area and finger count decouple.

HOW THE PIECES COMBINE INTO A CURRENT MISMATCH

At a FIXED gate voltage, each source moves the current independently:

    dI/I  =  (gm/ID).dVt  +  dbeta/beta  +  (dlnI/dlnW).dW/W
                                          +  (dlnI/dlnL).dL/L

and they are independent gaussians, so the sigmas add in quadrature.
The gm/ID in the first term is the whole of the bias dependence: the
Pelgrom coefficients themselves do not know what bias the device is at.
That is why an efficient device matches badly -- at gm/ID = 25 a
threshold shift is 25 times more current than it is at gm/ID = 1.

THE LENGTH TERM USES THE TABLE'S OWN DERIVATIVE

The textbook writes dI/I = -dL/L, because ID goes as W/L. That is not
the whole story: L also moves the threshold, through the short-channel
and reverse short-channel effects, and on this PDK there is a reverse
short-channel hump at L = 0.55 to 0.85 um where dVt/dL is anything but
small. dlnI/dlnL taken from the stored table at the actual bias contains
BOTH the geometric part and the threshold part, with no double counting,
because it is simply what the model says the current does when L moves.
So that is what is used, and the textbook -1 is reported beside it so the
difference is visible rather than hidden.

INPUT-REFERRED OFFSET

Dividing the current mismatch by gm/ID puts it back at the gate:

    sigma(dVos) = sqrt( sigma(dVt)^2 + (everything else)^2 / (gm/ID)^2 )

which is the number a differential pair is specified by. Note the first
term does not improve with bias at all -- only with area.

A PAIR IS sqrt(2) TIMES ONE DEVICE

Two devices mismatch independently, so the difference between them has
twice the variance. Every _PAIR quantity here is sqrt(2) times its
single-device counterpart. A current mirror and a differential pair are
both pairs in this sense.

WHAT IS A MODELLING CHOICE AND NOT A MEASUREMENT

The quadrature sum, and the claim that these four sources are
independent, are assumptions. They are the standard ones and they are
almost certainly right, but 'almost certainly' is not this project's
standard. Monte Carlo against the model measures them; until it has,
these numbers are REPORTED and are not allowed to size anything.
"""

import json
import os

import numpy as np


class Mismatch:
    """Pelgrom coefficients plus the arithmetic that applies them."""

    def __init__(self, lutdir):
        self.path = os.path.join(lutdir, "mismatch.json")
        self.d = None
        try:
            with open(self.path) as fh:
                self.d = json.load(fh)
        except Exception:                                    # noqa: BLE001
            self.d = None

    def __bool__(self):
        return self.d is not None

    @property
    def why_not(self):
        return (f"no mismatch.json beside the table. Build it with\n"
                f"    python3 make_mismatch.py <table>\n"
                f"  which reads the coefficients out of the PDK and "
                f"checks the formula.")

    # ---- the logarithmic derivative of ID with respect to L -----------
    def dlnI_dlnL(self, lut, L, VGS, VDS, VSB, h=0.02):
        """
        What the current does when the length moves, at a fixed bias.

        Central difference in ln L where the table allows it; one-sided
        at the ends, where the other side is off the grid and would come
        back NaN. Returns NaN only if neither side is available, which
        the caller must then report rather than paper over with -1.
        """
        def I(x):
            v = lut.eval("ID", L=np.array([x]), VGS=np.array([VGS]),
                         VDS=VDS, VSB=VSB)
            return float(np.asarray(v).ravel()[0])

        lo, hi = L * (1.0 - h), L * (1.0 + h)
        a, b, c = I(lo), I(hi), I(L)
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            return (np.log(b) - np.log(a)) / (np.log(hi) - np.log(lo))
        if np.isfinite(b) and b > 0 and c > 0:
            return (np.log(b) - np.log(c)) / (np.log(hi) - np.log(L))
        if np.isfinite(a) and a > 0 and c > 0:
            return (np.log(c) - np.log(a)) / (np.log(L) - np.log(lo))
        return float("nan")

    # ---- the sigmas at one sized device -------------------------------
    def at(self, lut, L, VGS, VDS, VSB, W_drawn, gmid):
        """
        Every mismatch quantity for this device. W_drawn and L are the
        DRAWN dimensions in metres, because that is what the model's own
        formula divides by -- not the effective width the rest of the
        table uses.
        """
        if not self:
            return {}
        d = self.d
        Wu, Lu = W_drawn * 1e6, L * 1e6
        area = Wu * Lu                               # um^2
        s_vt = d["A_VT"] / np.sqrt(area)             # V
        s_b = d["A_BETA"] / np.sqrt(area)            # fraction

        pL = self.dlnI_dlnL(lut, L, VGS, VDS, VSB)
        s_l = abs(pL) * (d["S_DL"] / L) if np.isfinite(pL) else float("nan")
        # dlnI/dlnW is 1 to within the narrow-width effect, and the term
        # is 3 nm in tens of microns: six parts per hundred thousand.
        # Measuring the exponent to improve it would be polishing a
        # number that cannot matter.
        s_w = d["S_DW"] / W_drawn

        geo2 = (0.0 if not np.isfinite(s_l) else s_l ** 2) + s_w ** 2
        rest2 = s_b ** 2 + geo2
        s_id = np.sqrt((gmid * s_vt) ** 2 + rest2)
        s_vos = np.sqrt(s_vt ** 2 + rest2 / gmid ** 2)
        r2 = np.sqrt(2.0)
        return {
            "SIGVT": s_vt, "SIGBETA": s_b * 100.0,
            "SIGW": s_w * 100.0, "SIGL": s_l * 100.0,
            "SIGID": s_id * 100.0, "SIGVOS": s_vos,
            "SIGID_PAIR": r2 * s_id * 100.0, "SIGVOS_PAIR": r2 * s_vos,
            "_area": area, "_dlnI_dlnL": pL,
            "_floor_pair": r2 * np.sqrt(geo2) * 100.0,
        }

    # ---- the part no width can remove ---------------------------------
    def floor_pair(self, lut, L, VGS, VDS, VSB):
        """
        The pair's current mismatch as the width goes to infinity.

        Everything that scales with area vanishes; what is left is the
        length term, because 3 nm of edge roughness on a 450 nm gate is
        the same 3 nm however wide the device is drawn.
        """
        if not self:
            return float("nan")
        pL = self.dlnI_dlnL(lut, L, VGS, VDS, VSB)
        if not np.isfinite(pL):
            return float("nan")
        return np.sqrt(2.0) * abs(pL) * (self.d["S_DL"] / L) * 100.0

    def L_for_floor(self, lut, target_pct, VGS, VDS, VSB):
        """
        The shortest stored L whose floor is below a target, so the tool
        can answer 'then how long does it have to be' instead of only
        saying no.
        """
        if not self:
            return None
        for Lv in np.asarray(lut.L, float):
            f = self.floor_pair(lut, float(Lv), VGS, VDS, VSB)
            if np.isfinite(f) and f < target_pct:
                return float(Lv)
        return None

    # ---- inverting the formula: what width meets a target -------------
    MM_KNOBS = ("SIGID", "SIGID_PAIR", "SIGVOS", "SIGVOS_PAIR")

    def _terms(self, knob, L, gmid, pL):
        """
        The three coefficients of whichever quantity is constrained.

            sigma^2  =  K/(Wu.Lu)  +  (cW/W)^2  +  F^2

        Every mismatch quantity has this shape: one term that shrinks
        with AREA, one that shrinks with width alone, and one that does
        not shrink at all. Only the coefficients differ, so one solver
        serves all four knobs. Returns (K, cW, F, r) with r the factor
        for a pair.
        """
        d = self.d
        r = 2.0 ** 0.5 if knob.upper().endswith("_PAIR") else 1.0
        F0 = abs(pL) * d["S_DL"] / L
        if knob.upper().startswith("SIGID"):
            return ((gmid * d["A_VT"]) ** 2 + d["A_BETA"] ** 2,
                    d["S_DW"], F0, r)
        # input-referred: divide the current mismatch by gm/ID, which
        # leaves the threshold term untouched -- that is the whole
        # reason an offset specification cannot be met by biasing
        return (d["A_VT"] ** 2 + (d["A_BETA"] / gmid) ** 2,
                d["S_DW"] / gmid, F0 / gmid, r)

    def floor_of(self, knob, L, gmid, pL):
        """The target below which no width succeeds, in the knob's units."""
        _, _, F, r = self._terms(knob, L, gmid, pL)
        f = r * F
        return f * 100.0 if knob.upper().startswith("SIGID") else f

    def W_for(self, target, knob, L, gmid, pL):
        """
        The DRAWN width at which a mismatch target is met, exactly.

        sigma^2 is a quadratic in 1/W, so this is a formula and not a
        search: no tolerance, no iteration count, no starting guess to
        get wrong. Mismatch is the only knob in the tool with a FLOOR,
        and a target at or below it has no solution at any width -- not
        a very large one, none -- so that case returns None and the
        caller is expected to say so rather than return a huge device.

        Note that mismatch is set by the DRAWN area, so this returns a
        drawn width where every other knob returns an effective one.
        """
        K, cW, F, r = self._terms(knob, L, gmid, pL)
        t = float(target) / r
        if knob.upper().startswith("SIGID"):
            t /= 100.0                      # the report works in per cent
        D = t * t - F * F
        if D <= 0.0:
            return None
        c1 = K / (L * 1e12)                 # K/(Wu.Lu) = c1/W, L in metres
        if cW <= 0.0:
            return c1 / D
        x = (-c1 + (c1 * c1 + 4.0 * cW * cW * D) ** 0.5) / (2.0 * cW * cW)
        return 1.0 / x if x > 0 else None
