#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | SIZING ASSISTANT -- the five-field engine
=============================================================================
How it is driven:

    a transistor's operating point is FIVE fields

        1  size        W      or  ID
        2  inversion   VGS    or  Vstar   (or gm/ID)
        3  length      L
        4              VDS
        5              VSB

    fill four, LEAVE ONE EMPTY, and the empty one becomes the sweep axis.
    Fill all five and there is nothing to sweep: the answer is one
    device, printed as a table of results.

Two typical uses:

  A  W, L, VDS, VSB given, VGS empty   ->  curves against VGS
       python3 gmid_sa.py lut_nmos_hv_tt_27 --W 40u --L 2u --VDS 1.5 --VSB 0 \\
               --y VSTAR,VOV --y ID --y GM --y GDS --cursor 817m

  B  ID, Vstar, VDS, VSB given, L empty ->  ro against L
       python3 gmid_sa.py lut_nmos_hv_tt_27 --ID 10u --VSTAR 200m --VDS 1.5 \\
               --VSB 0 --y RO --cursor 2u

  and all five given                    ->  one sized device
       python3 gmid_sa.py lut_nmos_hv_tt_27 --ID 10u --VSTAR 200m --L 2u \\
               --VDS 1.5 --VSB 0

RESULTS ARE ABSOLUTE. This is the difference from gmid_xplore.py: there the
charts are normalised to width and the current never enters; here a
current (or a width) is one of the inputs, so ro is in ohms and gm in
siemens FOR THAT DEVICE. Ask for ten times the current and ro is a tenth.

Numbers are typed SPICE style: 10u, 200m, 2u, 1.5,
5.6MEG. Lower-case m is milli; mega is MEG, as in SPICE.

DESIGN DECISIONS, said so rather than hidden:
    * the finger count. There is no finger field and the width is
      continuous, the way a designer reads it off a chart. Underneath,
      a width is that many
      reference fingers -- a fractional number of them -- because the
      table is exact for fingers of its own reference width and this
      project MEASURED that the current density moves when the finger
      width does. The result says how many reference fingers the width
      amounts to; the nearest whole number is the device to draw.
    * leaving TWO fields empty, or the size field empty. Refused: a
      2-D sweep is not implemented yet.
    * Ron is taken as VDS/ID.
"""

import argparse
import os
import re
import sys

import numpy as np

from gmid_size import Sizer

# ---- number entry, SPICE style ------------------------------------------
_SUF = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
        "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def num(txt):
    """'10u' -> 1e-5, '200m' -> 0.2, '5.6MEG' -> 5.6e6, '1.5' -> 1.5"""
    s = txt.strip()
    m = re.fullmatch(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
                     r"\s*([a-zA-Z]*)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"{txt!r} is not a number")
    v, suf = float(m.group(1)), m.group(2).lower()
    if suf == "":
        return v
    if suf not in _SUF:
        raise argparse.ArgumentTypeError(
            f"{txt!r}: unknown suffix {m.group(2)!r} "
            f"(use f p n u m k MEG G)")
    return v * _SUF[suf]


def eng(v, unit=""):
    """1e-5 -> '10u', 5.634e6 -> '5.634MEG': how the tool prints"""
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0" + unit
    for k, s in ((1e12, "T"), (1e9, "G"), (1e6, "MEG"), (1e3, "k"),
                 (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"),
                 (1e-12, "p"), (1e-15, "f"), (1e-18, "a")):
        if abs(v) >= k * 0.99995:
            return f"{v / k:.4g}{s}{unit}"
    return f"{v:.4g}{unit}"


# The Results table. Names on the left are what the window shows; the
# right is the expression this engine evaluates. Rows whose definition
# is a convention rather than a model quantity are marked.
RESULTS = [
    ("ID", "ID", "A"), ("W", "W", "m"), ("VGS", "VGS", "V"),
    ("L", "L", "m"), ("VDS", "VDS", "V"), ("VSB", "VSB", "V"),
    ("gm/ID", "GM/ID", "S/A"), ("Vstar", "VSTAR", "V"),
    ("fT", "FT", "Hz"), ("gm/gds", "GM/GDS", ""), ("VA", "VA", "V"),
    ("ID/W", "ID/W", "A/m"), ("gm/W", "GM/W", "S/m"),
    ("AREA", "W*L", "m^2"),
    ("gm", "GM", "S"), ("gmb", "GMB", "S"), ("gds", "GDS", "S"),
    ("ro", "RO", "ohm"),
    ("Ron", "VDS/ID", "ohm"),          # definition assumed
    ("VTH", "VTH", "V"), ("VDSAT", "VDSAT", "V"), ("Vov", "VOV", "V"),
    ("cgg", "CGG_TOT", "F"), ("cdd", "CDTOT", "F"),
    ("csg", "CSG", "F"), ("cdg", "CDG", "F"), ("cgd", "CGD", "F"),
    ("cgs", "CGS", "F"), ("cgb", "CGB", "F"),
    ("gamma", "GAMMA", ""), ("fknee", "FKNEE", "Hz"),
]
ALIAS = {n.upper(): e for n, e, _ in RESULTS}
UNIT = {n.upper(): u for n, _, u in RESULTS}


def split_expr(txt):
    """
    'Vov = VGS - VTH' -> ('Vov', 'VGS - VTH');  'gm/ID*fT' -> itself twice.
    A result name (ro, cgg, Vstar ...) becomes this engine's expression
    for it; anything else is upper-cased name by name, so gm/ID*fT and
    GM/ID*FT are the same thing, while sqrt(), log10() and the like are
    left alone.
    """
    txt = txt.strip()
    label, expr = txt, txt
    m = re.fullmatch(r"\s*([A-Za-z_][\w*]*)\s*=\s*(.+)", txt)
    if m and "==" not in txt:
        label, expr = m.group(1), m.group(2).strip()
    if expr.upper() in ALIAS:
        return label, ALIAS[expr.upper()]
    keep = {"sqrt", "abs", "log10", "exp", "pi", "np"}

    def up(mm):
        w = mm.group(0)
        if w in keep:
            return w
        return "(" + ALIAS[w.upper()] + ")" if (
            w.upper() in ALIAS and ALIAS[w.upper()] != w.upper()
            and "/" not in w) else w.upper()
    return label, re.sub(r"[A-Za-z_]\w*", up, expr)


def find_tables(root="."):
    """
    {device: {(corner, temp_C): directory}} for every main table under
    root. This is what stands behind the LUT / Corner / Temp selectors:
    This project keeps one directory per (corner, temperature), so
    the selector is rebuilt from each table's own meta.json -- never from
    its directory name.
    """
    import json
    from gmid_lut import corner_of_table
    out = {}
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "meta.json")
        if d.startswith("edge_") or not os.path.isfile(mp):
            continue
        try:
            with open(mp) as fh:
                m = json.load(fh)
            corner, _ = corner_of_table(m, d)
            dev = str(m["device"])
            temp = float(m.get("temp", 27.0))
        except Exception:                                    # noqa: BLE001
            continue
        out.setdefault(dev, {})[(corner.replace("mos_", "").upper(),
                                 temp)] = os.path.join(root, d)
    return out


class Assistant:
    """the five fields -> (L, VGS, VDS, VSB, W, ng) -> any quantity"""

    def __init__(self, lutdir):
        self.dir = lutdir
        self.siz = Sizer(lutdir)
        self.lut = self.siz.lut
        # one reference finger: the width at which the table is exact
        self.wf = self.lut.W_nom / self.lut.ng
        self.vmax = float(self.lut.meta["vmax"])

    # -- field 2: the gate voltage --------------------------------------
    def _vgs(self, inv, L, VDS, VSB):
        """VGS for arrays of points. NaN where the level is unreachable."""
        kind, val = inv
        val = np.broadcast_to(np.asarray(val, float), L.shape)
        if kind == "VGS":
            return val.astype(float).copy()
        gmid = 2.0 / val if kind == "VSTAR" else val
        # ONE contour solve for every point, whatever is being swept.
        # The solver works column by column and calls its constraint
        # with arrays shaped (..., n points), so VDS, VSB and the target
        # can each differ per column as long as they are broadcast the
        # same way. Solving point by point instead cost 30 ms a point:
        # sixteen seconds for one family of four curves against VDS.
        okp = np.isfinite(gmid) & (gmid > 0)
        tgt = np.where(okp, gmid, 1.0)

        def con(Lq, Vq):
            sh = np.shape(Lq)
            return self.lut.eval(
                "GM/ID", L=Lq, VGS=Vq, VDS=np.broadcast_to(VDS, sh),
                VSB=np.broadcast_to(VSB, sh)) / np.broadcast_to(tgt, sh)

        out = self.siz._vgs_curve(con, L, None, float(np.min(VSB)))
        # a gate voltage past the rated maximum at this VSB is not a
        # device the table holds
        out = np.where(okp & (out + VSB <= self.vmax + 1e-9), out, np.nan)
        return out

    def solve(self, size, inv, L, VDS, VSB):
        """
        The devices at these points, as arrays: L, VGS, VDS, VSB, W, ng.
        ng is FRACTIONAL -- the width counted in reference fingers.
        Rows that are not a device (unreachable, outside the table) hold
        NaN in VGS and W.
        """
        L, VDS, VSB = (np.atleast_1d(np.asarray(v, float))
                       for v in (L, VDS, VSB))
        n = max(L.size, VDS.size, VSB.size,
                np.size(inv[1]), np.size(size[1]))
        L, VDS, VSB = (np.broadcast_to(v, (n,)).astype(float)
                       for v in (L, VDS, VSB))
        g = self.lut.L
        inside = ((L >= g[0] * (1 - 1e-9)) & (L <= g[-1] * (1 + 1e-9))
                  & (VDS >= 0) & (VSB >= 0)
                  & (VDS + VSB <= self.vmax + 1e-9))
        vg = np.full(n, np.nan)
        if inside.any():
            iv = (inv[0], np.broadcast_to(np.asarray(inv[1], float),
                                          (n,))[inside])
            vg[inside] = self._vgs(iv, L[inside], VDS[inside], VSB[inside])
        ok = np.isfinite(vg)
        ng = np.full(n, np.nan)
        sv = np.broadcast_to(np.asarray(size[1], float), (n,))
        if size[0] == "W":
            ng[ok] = sv[ok] / self.wf
        elif ok.any():
            # A current. ID is exactly proportional to the count of
            # reference fingers, so one evaluation of the reference
            # device gives the count with no iteration.
            i = np.asarray(self.lut.eval(
                "ID", L=L[ok], VGS=vg[ok], VDS=VDS[ok], VSB=VSB[ok]),
                float)
            ng[ok] = np.where(i > 0, self.lut.ng * sv[ok] / i, np.nan)
        bad = ~(np.isfinite(ng) & (ng > 0))
        vg[bad] = np.nan
        ng[bad] = np.nan
        return dict(L=L, VGS=vg, VDS=VDS, VSB=VSB, W=ng * self.wf, ng=ng)

    def values(self, expr, d):
        """one quantity over a solve() result; NaN where there is no device"""
        _, e = split_expr(expr)
        ok = np.isfinite(d["VGS"])
        out = np.full(d["L"].shape, np.nan)
        if ok.any():
            whole = bool(np.all(np.abs(d["ng"][ok]
                                       - np.round(d["ng"][ok])) < 1e-9))
            out[ok] = np.asarray(self.lut.eval(
                e, L=d["L"][ok], VGS=d["VGS"][ok], VDS=d["VDS"][ok],
                VSB=d["VSB"][ok], W=d["W"][ok], ng=d["ng"][ok],
                parity=whole), float)
        return out

    # -- one point, for the all-five-given case and the cursor ----------
    def device(self, size, inv, L, VDS, VSB):
        d = self.solve(size, inv, L, VDS, VSB)
        if not np.isfinite(d["VGS"][0]):
            return None
        return d

    def value(self, name, d):
        try:
            return float(self.values(name, d)[0])
        except Exception:                                    # noqa: BLE001
            return np.nan

    # -- the sweep --------------------------------------------------------
    def axis(self, which, fixed):
        """the values the empty field is swept over: the table's own"""
        lut = self.lut
        if which == "L":
            g = np.asarray(lut.L, float)
            return np.exp(np.linspace(np.log(g[0]), np.log(g[-1]), 121))
        if which in ("VGS", "VDS"):
            top = self.vmax - (fixed.get("VSB") or 0.0)
            return np.linspace(0.0, top, 133)[1:]
        if which == "VSB":
            top = min(float(lut.VSB[-1]),
                      self.vmax - max(fixed.get("VDS") or 0.0, 0.0))
            return np.linspace(0.0, top, 67)
        if which == "VSTAR":
            return np.linspace(0.07, 0.6, 54)
        if which == "GMID":
            return np.linspace(3.0, 28.0, 51)
        raise ValueError(which)

    def at(self, which, xs, size, inv, f):
        """the devices when the swept field takes the values xs"""
        xs = np.atleast_1d(np.asarray(xs, float))
        L, VDS, VSB = f.get("L"), f.get("VDS"), f.get("VSB")
        if which == "L":
            L = xs
        elif which == "VDS":
            VDS = xs
        elif which == "VSB":
            VSB = xs
        else:
            inv = (which, xs)
        return self.solve(size, inv, L, VDS, VSB)

    def point(self, which, x, size, inv, f):
        d = self.at(which, [x], size, inv, f)
        return d if np.isfinite(d["VGS"][0]) else None

    def sweep(self, which, size, inv, f, names):
        xs = self.axis(which, f)
        d = self.at(which, xs, size, inv, f)
        out = {}
        for n in names:
            try:
                out[n] = self.values(n, d)
            except Exception:                                # noqa: BLE001
                out[n] = np.full(xs.shape, np.nan)
        return xs, out


def main():
    ap = argparse.ArgumentParser(
        description="five fields; the one left out is swept")
    ap.add_argument("lutdir")
    ap.add_argument("--W", type=num)
    ap.add_argument("--ID", type=num)
    ap.add_argument("--VGS", type=num)
    ap.add_argument("--VSTAR", type=num)
    ap.add_argument("--GMID", type=num)
    ap.add_argument("--L", type=num)
    ap.add_argument("--VDS", type=num)
    ap.add_argument("--VSB", type=num)
    ap.add_argument("--sweep", default=None,
                    choices=["VGS", "VSTAR", "GMID"],
                    help="when the inversion field is the empty one: "
                         "which of its three forms is swept (default VGS)")
    ap.add_argument("--y", action="append", default=[],
                    help="what to plot; repeat for more plots, comma for "
                         "several traces on one plot. Names as in the "
                         "results table, or any expression")
    ap.add_argument("--cursor", type=num, default=None,
                    help="read every plotted quantity at this x")
    ap.add_argument("--out", default="sa")
    a = ap.parse_args()

    # ---- the five fields ------------------------------------------------
    if a.W is not None and a.ID is not None:
        ap.error("W and ID are the SAME field (the size); give one")
    ninv = sum(v is not None for v in (a.VGS, a.VSTAR, a.GMID))
    if ninv > 1:
        ap.error("VGS, VSTAR and GMID are the SAME field (the inversion "
                 "level); give one")
    size = (("W", a.W) if a.W is not None else
            ("ID", a.ID) if a.ID is not None else None)
    inv = (("VGS", a.VGS) if a.VGS is not None else
           ("VSTAR", a.VSTAR) if a.VSTAR is not None else
           ("GMID", a.GMID) if a.GMID is not None else None)
    f = dict(L=a.L, VDS=a.VDS, VSB=a.VSB)
    empty = ([] if size else ["size"]) + ([] if inv else ["inversion"]) \
        + [k for k in ("L", "VDS", "VSB") if f[k] is None]
    if len(empty) > 1:
        ap.error(f"{len(empty)} fields are empty ({', '.join(empty)}). "
                 f"Exactly one may be: it becomes the sweep. Sweeping "
                 f"two at once is not supported.")
    if empty == ["size"]:
        ap.error("the size field (W or ID) is empty. Sweeping the size is "
                 "not supported.")

    sa = Assistant(a.lutdir)
    lut = sa.lut
    print("=" * 72)
    print(f"  SIZING ASSISTANT   {a.lutdir}   ({lut.device}, "
          f"T = {lut.temp_c:g} C)")
    print("  " + "   ".join(
        [f"{size[0]} = {eng(size[1])}"]
        + [f"{inv[0]} = {eng(inv[1])}" if inv else "inversion = (swept)"]
        + [f"{k} = {eng(f[k]) if f[k] is not None else '(swept)'}"
           for k in ("L", "VDS", "VSB")]))
    print("=" * 72)

    # ---- all five given: one device -------------------------------------
    if not empty:
        d = sa.device(size, inv, f["L"], f["VDS"], f["VSB"])
        if d is None:
            sys.exit("  no such device: this inversion level cannot be "
                     "reached at this L, VDS and VSB, or the bias is "
                     "outside the table.")
        print(f"  fingers : this width is {d['ng'][0]:.2f} fingers of the "
              f"table's reference {eng(sa.wf)} finger")
        print(f"\n  {'#':>3}  {'Name':<8} {'Value':>12}")
        print("  " + "-" * 28)
        for i, (n, _e, u) in enumerate(RESULTS, 1):
            v = sa.value(n, d)
            if np.isfinite(v):
                print(f"  {i:>3}  {n:<8} {eng(v):>12}  {u}")
        acc = sa.siz.l_accuracy_at(float(d["L"][0]))
        if acc is not None:
            print(f"\n  table accuracy at this length: up to {acc[0]:.3g} %"
                  f"  ({acc[1]})")
        return 0

    # ---- one empty: the sweep -------------------------------------------
    which = empty[0]
    if which == "inversion":
        which = a.sweep or "VGS"
    plots = [[t.strip() for t in y.split(",") if t.strip()]
             for y in a.y] or [["RO"]]
    names = sorted({n for p in plots for n in p})
    xs, ys = sa.sweep(which, size, inv, f, names)
    ok = np.isfinite(np.vstack([ys[n] for n in names])).any(axis=0)
    print(f"  swept   : {which}   {eng(xs[0])} .. {eng(xs[-1])}   "
          f"({int(ok.sum())}/{len(xs)} points reachable)")

    os.makedirs(a.out, exist_ok=True)
    csv = os.path.join(a.out, "sweep.csv")
    with open(csv, "w") as fh:
        fh.write(",".join([which] + names) + "\n")
        for i, x in enumerate(xs):
            fh.write(",".join([f"{x:.8g}"] + [
                "" if not np.isfinite(ys[n][i]) else f"{ys[n][i]:.8g}"
                for n in names]) + "\n")

    # ---- the cursor: x typed in, every y read off the curve -------------
    cur = None
    if a.cursor is not None:
        d = sa.point(which, a.cursor, size, inv, f)
        print(f"\n  CURSOR   {which} = {eng(a.cursor)}")
        if d is None:
            print("    no device there: outside what this transistor "
                  "reaches with the other four fields as given")
        else:
            cur = {n: sa.value(n, d) for n in names}
            for n in names:
                print(f"    {n:<10} = {eng(cur[n])}"
                      f"  {UNIT.get(n.upper(), '')}")
            print(f"    (device there: W = {eng(d['W'][0])} = "
                  f"{d['ng'][0]:.2f} reference fingers, "
                  f"L = {eng(d['L'][0])}, VGS = {eng(d['VGS'][0])})")
            acc = sa.siz.l_accuracy_at(float(d["L"][0]))
            if acc is not None:
                print(f"    table accuracy at this length: up to "
                      f"{acc[0]:.3g} %  ({acc[1]})")

    # ---- the figure: up to four plots, 2 x 2, as in the tool ------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n  wrote {csv}  (matplotlib not installed: no picture)")
        return 0
    n = len(plots)
    rows, cols = (1, 1) if n == 1 else (1, 2) if n == 2 else \
        (2, 2) if n <= 4 else ((n + 2) // 3, 3)
    fig, axs = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.0 * rows),
                            squeeze=False)
    par = (f"{inv[0]}={eng(inv[1])}" if inv and which != "L"
           and f["L"] is None else
           f"L={eng(f['L'])}" if f["L"] is not None else
           f"{inv[0]}={eng(inv[1])}")
    for k, p in enumerate(plots):
        ax = axs[k // cols][k % cols]
        for j, nme in enumerate(p):
            ax.plot(xs, ys[nme], lw=1.5, ls=("-", ":", "--", "-.")[j % 4],
                    label=nme)
        ax.set_xlabel(which)
        ax.set_ylabel(", ".join(p))
        ax.set_title(f"Plot 1{'ABCDEFGHI'[k]} ({a.lutdir})", fontsize=9,
                     loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if cur is not None and np.isfinite(cur[p[0]]):
            txt = "\n".join([f"{which}={eng(a.cursor)}"]
                            + [f"{q}={eng(cur[q])}" for q in p] + [par])
            ax.annotate(txt, (a.cursor, cur[p[0]]), xytext=(25, 25),
                        textcoords="offset points", fontsize=8, ha="left",
                        bbox=dict(boxstyle="round", fc="#f4c7cf",
                                  ec="#b06070"),
                        arrowprops=dict(arrowstyle="->", color="k"))
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")
    fig.tight_layout()
    png = os.path.join(a.out, "figure.png")
    fig.savefig(png, dpi=140)
    print(f"\n  wrote {png}  and  {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
