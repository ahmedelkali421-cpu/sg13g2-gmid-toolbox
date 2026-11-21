#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | SIZING ASSISTANT -- the window
=============================================================================
    python3 gmid_gui.py                run it in the directory of the tables
    python3 gmid_gui.py /path/to/tables

The layout:

    centre   Figure tabs; each figure is a 2 x 2 grid of plots (A B C D);
             under every plot: the x value of its cursor (typed, stepped
             or clicked), the Parameter that picks the curve of a family,
             and a read-only y readout
    right    Plot A/B/C/D selector, Import (copy another plot's
             settings), LUT / Corner / Temp, the FIVE FIELDS, the Results
             table and the Y-Expr box
    bottom   the log

How it is driven:

    fill four of the five fields and leave ONE empty. The empty one is
    the sweep. Press Plot beside a result (or type a Y-Expr) and it is
    drawn against that field in the selected plot. Fill all five and the
    Results column shows numbers: one sized device.

    several values in one field, separated by commas (L = 0.5u,1u,2u),
    give a family of curves; so does ticking All beside Corner or Temp.

    click on a plot, or type an x under it: the cursor moves there and
    every trace is read AT that x from the table itself -- not from the
    drawn polyline, so it is not limited to the points that were drawn.

    THE MAGIC CURSOR. The Parameter box is not only a choice among the
    drawn curves: type any value in it (L = 220n while the curves are
    180n, 400n ...) and the cursor reads there, BETWEEN the curves. Or
    type the y you want in the readout and the Parameter is solved for.
    With a single curve drawn, a typed y moves the cursor along it.

All the numbers come from gmid_sa.py, the same engine the command line
uses; this file computes nothing of its own. Not implemented yet:
stacked devices, a frequency field, and sweeping two fields at once.
"""

import datetime
import itertools
import json
import math
import os
import sys

import numpy as np

import matplotlib
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gmid_sa import Assistant, RESULTS, UNIT, eng, find_tables, num, split_expr

SIZE_KINDS = ["W", "ID"]
INV_KINDS = ["VGS", "Vstar", "gm/ID"]
INV_KEY = {"VGS": "VGS", "Vstar": "VSTAR", "gm/ID": "GMID"}
SLOTS = "ABCD"
STYLES = ["-", ":", "--", "-."]


def _enum(owner, scope, name):
    """Qt5 spells it owner.name, Qt6 owner.scope.name; take whichever"""
    s = getattr(owner, scope, None)
    return getattr(s, name) if s is not None and hasattr(s, name) \
        else getattr(owner, name)


def parse_list(txt):
    """
    '' -> [];  '2u' -> [2e-6];  '0.5u, 1u,2u' -> three values;
    '180n, 400n:200n:1u' -> 180n, 400n, 600n, 800n, 1u

    The start:step:stop form gives evenly spaced values, so a family
    of five lengths is one short entry.
    """
    out = []
    for t in txt.replace(";", ",").split(","):
        t = t.strip()
        if not t:
            continue
        if ":" in t:
            a, st, b = (num(x) for x in t.split(":"))
            if st <= 0 or b < a:
                raise ValueError(f"{t!r}: a range is start:step:stop with "
                                 f"a positive step")
            n = int(np.floor((b - a) / st * (1 + 1e-9) + 1e-9)) + 1
            out += [a + i * st for i in range(n)]
        else:
            out.append(num(t))
    return out


def blank_state(dev="", corner="TT", temp=27.0):
    return dict(dev=dev, devs=[], corner=corner, temp=temp,
                all_corner=False, all_temp=False, size_kind="ID", size="",
                inv_kind="Vstar", inv="", L="", VDS="", VSB="0", traces=[],
                cursor=None, fam=0, pval=None, logx=False, logy=False)


class Engine:
    """tables on disk -> Assistant objects, loaded once each"""

    def __init__(self, root):
        self.root = root
        self.tables = find_tables(root)
        self._sa = {}

    def sa(self, path):
        if path not in self._sa:
            self._sa[path] = Assistant(path)
        return self._sa[path]

    def corners(self, dev):
        return sorted({c for c, _ in self.tables.get(dev, {})})

    def temps(self, dev):
        return sorted({t for _, t in self.tables.get(dev, {})})

    def fields(self, st):
        """
        The five fields of a state, parsed:
        (size, inv, fixed, which, par) or raises ValueError with the
        sentence to show. `par` is (field name, [values]) when one field
        holds a list -- the Parameter of the family -- else None.
        """
        vals = dict(size=parse_list(st["size"]), inv=parse_list(st["inv"]),
                    L=parse_list(st["L"]), VDS=parse_list(st["VDS"]),
                    VSB=parse_list(st["VSB"]))
        empty = [k for k, v in vals.items() if not v]
        multi = [k for k, v in vals.items() if len(v) > 1]
        if len(empty) > 1:
            names = dict(size=st["size_kind"], inv=st["inv_kind"])
            raise ValueError(
                "fields empty: " + ", ".join(names.get(k, k) for k in empty)
                + ". Leave exactly ONE empty -- it becomes the sweep.")
        if "size" in empty:
            raise ValueError(f"{st['size_kind']} is empty. The size field "
                             f"cannot be the sweep; give W or ID.")
        if len(multi) > 1:
            raise ValueError("more than one field holds a list. Only one "
                             "can be the Parameter of a family.")
        which = None
        if empty:
            which = INV_KEY[st["inv_kind"]] if empty[0] == "inv" \
                else empty[0]
        par = None
        if multi:
            k = multi[0]
            nm = (st["size_kind"] if k == "size" else
                  st["inv_kind"] if k == "inv" else k)
            par = (k, nm, vals[k])
        return vals, which, par

    def families(self, st):
        """
        Every curve family member of a state:
        [dict(label, sa, size, inv, f)], plus which field is swept.
        A member is one (table, parameter value) pair.
        """
        vals, which, par = self.fields(st)
        # Device Comparison: st["devs"] holds more than one device and
        # each becomes its own curve, the SAME settings on every one. An
        # empty or single list is the ordinary single-device case, so the
        # Sizing Assistant and the comparison share this one path and
        # cannot drift apart.
        devs = [d for d in (st.get("devs") or []) if d in self.tables] \
            or [st["dev"]]
        many_dev = len(devs) > 1
        members = []
        pvals = par[2] if par else [None]
        for dev in devs:
            tabs = self.tables.get(dev, {})
            cs = self.corners(dev) if st["all_corner"] else [st["corner"]]
            ts = self.temps(dev) if st["all_temp"] else [st["temp"]]
            for c, t, pv in itertools.product(cs, ts, pvals):
                path = tabs.get((c, t))
                if path is None:
                    continue
                v = {k: (x[0] if x else None) for k, x in vals.items()}
                if par:
                    v[par[0]] = pv
                size = (st["size_kind"], v["size"])
                inv = (INV_KEY[st["inv_kind"]], v["inv"]) \
                    if v["inv"] is not None else None
                bits = []
                if many_dev:
                    bits.append(dev)
                if par:
                    bits.append(f"{par[1]}={eng(pv)}")
                if st["all_corner"]:
                    bits.append(c)
                if st["all_temp"]:
                    bits.append(f"{t:g}C")
                # Which field the Parameter box under the plot stands
                # for: the one holding a list, else L, else the
                # inversion level when L is the sweep. When devices are compared the Parameter box
                # instead steps between the devices.
                if many_dev and not par:
                    pkey, pname = "dev", "device"
                elif par:
                    pkey, pname = par[0], par[1]
                else:
                    pkey, pname = next(
                        ((k, nm) for k, nm in (("L", "L"),
                                               ("inv", st["inv_kind"]))
                         if v[k] is not None), (None, ""))
                    if pkey and not bits:
                        bits.append(f"{pname}={eng(v[pkey])}")
                members.append(dict(
                    label=" ".join(bits), sa=self.sa(path), size=size,
                    inv=inv, f=dict(L=v["L"], VDS=v["VDS"], VSB=v["VSB"]),
                    pkey=pkey, pname=pname, dev=dev,
                    pv=(v[pkey] if pkey and pkey != "dev" else None)))
        if not members:
            raise ValueError(f"no table for {st['dev']} at corner "
                             f"{st['corner']}, {st['temp']:g} C")
        return members, which


class PlotCell(QtWidgets.QFrame):
    """one of the four plots: canvas + the three boxes under it"""

    def __init__(self, win, name):
        super().__init__()
        self.win, self.name = win, name
        self.st = blank_state()
        self.members, self.which = [], None
        self.xs, self.ys = None, []       # ys[member][trace] = array
        self.setFrameShape(_enum(QtWidgets.QFrame, "Shape", "StyledPanel"))
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        # dpi stated, not inherited: a machine whose matplotlibrc sets
        # figure.dpi to 200 drew every label at twice the size and
        # squeezed the axes flat
        self.fig = Figure(figsize=(5, 3.2), dpi=100, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        v.addWidget(self.canvas, 1)
        h = QtWidgets.QHBoxLayout()
        self.xlab = QtWidgets.QLabel("X-axis")
        self.xedit = QtWidgets.QLineEdit()
        self.xedit.setFixedWidth(80)
        self.xedit.returnPressed.connect(self.on_xedit)
        up, dn = QtWidgets.QToolButton(), QtWidgets.QToolButton()
        up.setText("▲")
        dn.setText("▼")
        up.clicked.connect(lambda: self.step(+1))
        dn.clicked.connect(lambda: self.step(-1))
        self.plab = QtWidgets.QLabel("Parameter")
        self.pbox = QtWidgets.QComboBox()
        self.pbox.setMinimumWidth(120)
        self.pbox.setEditable(True)
        self.pbox.setInsertPolicy(_enum(QtWidgets.QComboBox,
                                        "InsertPolicy", "NoInsert"))
        self.pbox.activated.connect(self.on_param)
        self.pbox.lineEdit().returnPressed.connect(self.on_param_typed)
        self.pbox.setToolTip(
            "pick a drawn curve, or TYPE any value (e.g. 220n): the "
            "cursor then reads BETWEEN the curves")
        self.ylab = QtWidgets.QLabel("Y-axis")
        self.yout = QtWidgets.QLineEdit()
        self.yout.setFixedWidth(95)
        self.yout.returnPressed.connect(self.on_ytyped)
        self.yout.setToolTip(
            "the reading at the cursor. TYPE a value and the Parameter "
            "is solved so that the first trace equals it at this x")
        self.lx = QtWidgets.QCheckBox("log x")
        self.ly = QtWidgets.QCheckBox("log y")
        self.lx.toggled.connect(self.on_log)
        self.ly.toggled.connect(self.on_log)
        self.vbtn = QtWidgets.QToolButton()
        self.vbtn.setText("Verify")
        self.vbtn.setToolTip(
            "run ngspice ONCE on the device under the cursor and show "
            "table vs simulator. Writes cursor_exact.spice so you can "
            "re-run it yourself.")
        self.vbtn.clicked.connect(self.on_verify)
        clip = QtWidgets.QToolButton()
        clip.setText("⧉")
        clip.setToolTip("copy this chart to the clipboard")
        clip.clicked.connect(self.to_clipboard)
        for w in (self.xlab, self.xedit, up, dn, self.plab, self.pbox,
                  self.ylab, self.yout):
            h.addWidget(w)
        h.addStretch(1)
        for w in (self.lx, self.ly, self.vbtn, clip):
            h.addWidget(w)
        v.addLayout(h)
        self.redraw()

    # ---- selection ------------------------------------------------------
    def mousePressEvent(self, ev):
        self.win.select(self)
        super().mousePressEvent(ev)

    def mark(self, on):
        self.setStyleSheet("PlotCell{border:2px solid #4a7fb5;}" if on
                           else "PlotCell{border:1px solid #b0b0b0;}")

    # ---- computing ------------------------------------------------------
    def compute(self):
        """re-solve everything from self.st; raises ValueError to report"""
        members, which = self.win.eng.families(self.st)
        if which is None:
            raise ValueError("all five fields are filled, so there is "
                             "nothing to sweep: the Results column shows "
                             "this one device. Empty a field to plot.")
        xs = members[0]["sa"].axis(which, members[0]["f"])
        ys = []
        for m in members:
            d = m["sa"].at(which, xs, m["size"], m["inv"], m["f"])
            row = []
            for t in self.st["traces"]:
                try:
                    row.append(m["sa"].values(t, d))
                except Exception as exc:                     # noqa: BLE001
                    raise ValueError(f"cannot evaluate {t!r}: {exc}")
            ys.append(row)
        # EVERY point unreachable: an empty plot tells the user nothing,
        # so name the most likely cause -- a bias beyond the device's
        # rated voltage (VDS + VSB > its maximum), which is what an LV
        # table does with VSB = 2.
        import numpy as _np
        if not any(_np.isfinite(_np.asarray(y)).any()
                   for row in ys for y in row):
            m0 = members[0]
            vmax = float(m0["sa"].lut.meta["vmax"])
            vds = m0["f"]["VDS"] or 0.0
            vsb = m0["f"]["VSB"] or 0.0
            why = ""
            if which not in ("VDS", "VSB") and vds + vsb > vmax + 1e-9:
                why = (f": VDS + VSB = {eng(vds + vsb)}V is above this "
                       f"device's rating of {eng(vmax)}V. Lower VSB or "
                       f"VDS.")
            raise ValueError(f"nothing to draw -- no point is reachable"
                             + (why or ". The bias is outside the table "
                                      "everywhere along the sweep."))
        if self.members and members and (
                self.members[0]["pkey"] != members[0]["pkey"]):
            self.st["pval"] = None
        self.members, self.which, self.xs, self.ys = members, which, xs, ys
        self.st["fam"] = min(self.st["fam"], len(members) - 1)
        if self.st["cursor"] is None or not (
                xs[0] <= self.st["cursor"] <= xs[-1]):
            self.st["cursor"] = None

    # ---- the off-trace cursor -------------------------------------------
    # Curves drawn for L = 180n, 400n, 600n, 800n, 1u, and the cursor
    # can still read gm/gds at L = 220n -- a length that is NOT one of
    # the curves, with its annotation floating in the white space
    # between two of them. That is the "magic cursor": the Parameter
    # under the plot is a free value, not a choice among the drawn
    # traces.
    def member(self):
        return self.members[self.st["fam"]]

    def pvalue(self):
        """the Parameter the cursor reads at: typed, else the curve's"""
        pv = self.st.get("pval")
        return self.member()["pv"] if pv is None else pv

    def devices_at(self, x, pvals):
        """
        the devices at cursor x for an ARRAY of Parameter values, on the
        current member's table. One vectorised solve.
        """
        m = self.member()
        pvals = np.atleast_1d(np.asarray(pvals, float))
        size, inv, f = m["size"], m["inv"], dict(m["f"])
        k = m["pkey"]
        if k == "size":
            size = (size[0], pvals)
        elif k == "inv":
            inv = (inv[0], pvals)
        elif k is not None:
            f[k] = pvals
        w = self.which
        if w in ("L", "VDS", "VSB"):
            f[w] = x
        else:
            inv = (w, x)
        return m["sa"].solve(size, inv, f["L"], f["VDS"], f["VSB"])

    def read_cursor(self):
        """the traces AT the cursor, from the engine itself"""
        x = self.st["cursor"]
        if x is None or not self.members:
            return None
        m = self.member()
        pv = self.pvalue()
        d = self.devices_at(x, [pv if pv is not None else 0.0])
        if not np.isfinite(d["VGS"][0]):
            return None
        return [m["sa"].value(t, d) for t in self.st["traces"]]

    def plabel(self):
        m = self.member()
        if m["pkey"] is None:
            return m["label"]
        if self.st.get("pval") is None:
            return m["label"]
        rest = [b for b in m["label"].split(" ")
                if not b.startswith(m["pname"] + "=")]
        return " ".join([f"{m['pname']}={eng(self.st['pval'])}"] + rest)

    def prange(self):
        """where the Parameter may be searched when y is typed"""
        m = self.member()
        sa, k = m["sa"], m["pkey"]
        if k == "L":
            g = sa.lut.L
            return np.exp(np.linspace(np.log(g[0]), np.log(g[-1]), 241))
        if k in ("VDS", "VSB"):
            return np.linspace(0.0, sa.vmax, 199)[1:]
        if k == "inv":
            kind = m["inv"][0]
            return (np.linspace(0.06, 0.8, 223) if kind == "VSTAR" else
                    np.linspace(2.0, 30.0, 225) if kind == "GMID" else
                    np.linspace(0.0, sa.vmax, 265)[1:])
        return None

    def solve_param(self, ytarget):
        """
        every Parameter value at which the FIRST trace equals ytarget at
        the cursor's x. Scanned, then bisected, so that two answers are
        both found (a quantity need not be monotone in L).
        """
        grid = self.prange()
        x = self.st["cursor"]
        if grid is None or x is None or ytarget == 0:
            return None
        m, t = self.member(), self.st["traces"][0]

        def f(p):
            return m["sa"].values(t, self.devices_at(x, p)) / ytarget - 1.0
        v = f(grid)
        roots = []
        for i in range(len(grid) - 1):
            if not (np.isfinite(v[i]) and np.isfinite(v[i + 1])) \
                    or v[i] * v[i + 1] > 0:
                continue
            a, b, fa = grid[i], grid[i + 1], v[i]
            for _ in range(40):
                mid = 0.5 * (a + b)
                fm = float(f([mid])[0])
                if not np.isfinite(fm):
                    break
                if fa * fm <= 0:
                    b = mid
                else:
                    a, fa = mid, fm
            roots.append(0.5 * (a + b))
        fin = v[np.isfinite(v)]
        span = ((fin.min() + 1) * ytarget, (fin.max() + 1) * ytarget) \
            if fin.size else None
        return roots, span

    def solve_x(self, ytarget):
        """
        every x at which the FIRST trace equals ytarget, ALONG the curve
        the cursor is on (its Parameter, typed or drawn). Scanned on the
        plotted x grid, then bisected on the engine itself.

        This is what a typed y means when there is no family to choose
        from: one curve, so the only free quantity left is x. Example:
        ID, gm/ID, VDS, VSB fixed, x = L, y = gm/gds; typing 50 must
        return the L that gives gm/gds = 50, not hold L and look for a
        gm/ID that does.
        """
        x0 = self.st["cursor"]
        if self.xs is None or ytarget == 0:
            return None
        m, t = self.member(), self.st["traces"][0]
        pv = self.pvalue()
        pv = 0.0 if pv is None else pv

        def f(x):
            d = self.devices_at(float(x), [pv])
            return float(m["sa"].values(t, d)[0]) / ytarget - 1.0
        xs = np.asarray(self.xs, float)
        if self.st.get("pval") is None:
            v = np.asarray(self.ys[self.st["fam"]][0], float) / ytarget - 1.0
        else:
            v = np.array([f(x) for x in xs])
        logx = bool(self.st.get("logx")) and xs[0] > 0
        roots = []
        for i in range(len(xs) - 1):
            if not (np.isfinite(v[i]) and np.isfinite(v[i + 1])) \
                    or v[i] * v[i + 1] > 0:
                continue
            a, b, fa = xs[i], xs[i + 1], v[i]
            for _ in range(50):
                mid = math.sqrt(a * b) if logx else 0.5 * (a + b)
                fm = f(mid)
                if not np.isfinite(fm):
                    break
                if fa * fm <= 0:
                    b = mid
                else:
                    a, fa = mid, fm
            roots.append(math.sqrt(a * b) if logx else 0.5 * (a + b))
        fin = v[np.isfinite(v)]
        span = ((fin.min() + 1) * ytarget, (fin.max() + 1) * ytarget) \
            if fin.size else None
        if x0 is not None and roots:
            roots.sort(key=lambda r: abs(r - x0))
        return roots, span

    # ---- drawing --------------------------------------------------------
    def redraw(self):
        ax = self.ax
        ax.clear()
        st = self.st
        ax.set_title(f"Plot {self.name}"
                     + (f" ({st['dev']})" if self.members else ""),
                     fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        self.pbox.blockSignals(True)
        self.pbox.clear()
        if not self.members or not st["traces"]:
            self.xlab.setText("X-axis")
            self.plab.setText("Parameter")
            self.ylab.setText("Y-axis")
            self.xedit.setText("")
            self.yout.setText("")
            self.pbox.blockSignals(False)
            self.canvas.draw_idle()
            return
        labels = [split_expr(t)[0] for t in st["traces"]]
        many = len(self.members) > 1
        for i, m in enumerate(self.members):
            for j, lab in enumerate(labels):
                ax.plot(self.xs, self.ys[i][j], lw=1.5,
                        ls=STYLES[j % 4],
                        color=f"C{i if many else j}",
                        label=lab + (f"  {m['label']}" if many else ""))
        ax.set_xlabel(self.which)
        ax.set_ylabel(", ".join(labels), fontsize=9)
        if st["logx"]:
            ax.set_xscale("log")
        if st["logy"]:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        self.pbox.addItems([m["label"] for m in self.members])
        self.pbox.setCurrentIndex(st["fam"])
        if st.get("pval") is not None:
            self.pbox.setEditText(self.plabel())
        self.pbox.blockSignals(False)
        self.xlab.setText(self.which)
        self.plab.setText("Parameter" if many else "")
        self.ylab.setText(labels[0])
        vals = self.read_cursor()
        if vals is None:
            self.xedit.setText("")
            self.yout.setText("")
        else:
            x = st["cursor"]
            self.xedit.setText(eng(x))
            self.yout.setText(eng(vals[0]))
            pl = self.plabel()
            lines = [f"{self.which}={eng(x)}"] \
                + [f"{l}={eng(v)}" for l, v in zip(labels, vals)] \
                + ([pl] if pl else [])
            if np.isfinite(vals[0]):
                ax.annotate("\n".join(lines), (x, vals[0]),
                            xytext=(22, 22), textcoords="offset points",
                            fontsize=8,
                            bbox=dict(boxstyle="round", fc="#f4c7cf",
                                      ec="#b06070"),
                            arrowprops=dict(arrowstyle="->", color="k"))
        self.canvas.draw_idle()

    # ---- the cursor -----------------------------------------------------
    def set_cursor(self, x):
        if self.xs is None:
            return
        self.st["cursor"] = float(min(max(x, self.xs[0]), self.xs[-1]))
        self.redraw()
        vals = self.read_cursor()
        if vals is not None:
            labs = [split_expr(t)[0] for t in self.st["traces"]]
            self.win.log(f"Plot {self.name}: {self.which}="
                         f"{eng(self.st['cursor'])}  "
                         + "  ".join(f"{l}={eng(v)}"
                                     for l, v in zip(labs, vals)))

    def on_click(self, ev):
        self.win.select(self)
        if ev.inaxes is self.ax and ev.xdata is not None:
            self.set_cursor(ev.xdata)

    def on_xedit(self):
        try:
            self.set_cursor(num(self.xedit.text()))
        except Exception:                                    # noqa: BLE001
            self.win.log(f"'{self.xedit.text()}' is not a number")

    def step(self, sign):
        if self.xs is None:
            return
        x = self.st["cursor"]
        x = self.xs[len(self.xs) // 2] if x is None else x
        self.set_cursor(x + sign * (self.xs[-1] - self.xs[0]) / 200.0)

    def on_param(self, i):
        self.st["fam"] = int(i)
        self.st["pval"] = None            # back onto a drawn curve
        self.redraw()

    def on_param_typed(self):
        """a typed Parameter: the cursor leaves the drawn curves"""
        if not self.members:
            return
        txt = self.pbox.currentText().strip()
        labs = [m["label"] for m in self.members]
        if txt in labs:
            return self.on_param(labs.index(txt))
        m = self.member()
        if m["pkey"] is None:
            return
        try:
            v = num(txt.split(" ")[0].split("=")[-1])
        except Exception:                                    # noqa: BLE001
            self.win.log(f"Plot {self.name}: '{txt}' is not a value of "
                         f"{m['pname']}")
            return self.redraw()
        self.st["pval"] = float(v)
        self.redraw()
        vals = self.read_cursor()
        if vals is None:
            self.win.log(f"Plot {self.name}: no device at "
                         f"{m['pname']}={eng(v)} (outside the table, or "
                         f"unreachable at this x)")
        else:
            labs = [split_expr(t)[0] for t in self.st["traces"]]
            self.win.log(f"Plot {self.name}: OFF-TRACE  {self.which}="
                         f"{eng(self.st['cursor'])}  {m['pname']}={eng(v)}"
                         f"  " + "  ".join(f"{l}={eng(y)}"
                                           for l, y in zip(labs, vals)))

    def on_ytyped(self):
        """
        a typed y. With a FAMILY of curves (several Parameter values
        drawn) it is the magic cursor: hold x, solve the Parameter. With
        ONE curve there is no family to move through, so it moves the
        cursor along the curve to the x that gives y.
        """
        if not self.members:
            return
        m = self.member()
        lab = split_expr(self.st["traces"][0])[0]
        try:
            y = num(self.yout.text())
        except Exception:                                    # noqa: BLE001
            self.win.log(f"'{self.yout.text()}' is not a number")
            return self.redraw()
        if len(self.members) == 1:
            res = self.solve_x(y)
            if res is None:
                return self.redraw()
            roots, span = res
            if not roots:
                self.win.log(
                    f"Plot {self.name}: {lab}={eng(y)} is not reached on "
                    f"this curve" + (f"; it spans {eng(span[0])} .. "
                                     f"{eng(span[1])}" if span else ""))
                return self.redraw()
            self.st["cursor"] = float(roots[0])
            self.redraw()
            self.win.log(
                f"Plot {self.name}: {lab}={eng(y)}  ->  "
                f"{self.which}={eng(roots[0])}"
                + (f"   ({len(roots)} solutions: "
                   + ", ".join(eng(r) for r in roots)
                   + "; nearest to the cursor taken)"
                   if len(roots) > 1 else ""))
            return
        if self.st["cursor"] is None:
            return
        res = self.solve_param(y)
        if res is None:
            self.win.log(f"Plot {self.name}: the Parameter here "
                         f"({m['pname'] or 'none'}) cannot be solved for")
            return self.redraw()
        roots, span = res
        if not roots:
            self.win.log(
                f"Plot {self.name}: NO {m['pname']} gives {lab}={eng(y)} "
                f"at {self.which}={eng(self.st['cursor'])}"
                + (f"; reachable there: {eng(span[0])} .. {eng(span[1])}"
                   if span else ""))
            return self.redraw()
        cur = self.pvalue()
        pick = min(roots, key=lambda r: abs(np.log(r / cur))
                   if cur and cur > 0 and r > 0 else abs(r - (cur or 0)))
        self.st["pval"] = float(pick)
        self.redraw()
        self.win.log(
            f"Plot {self.name}: {lab}={eng(y)} at {self.which}="
            f"{eng(self.st['cursor'])}  ->  {m['pname']}={eng(pick)}"
            + (f"   ({len(roots)} solutions: "
               + ", ".join(eng(r) for r in roots) + "; nearest taken)"
               if len(roots) > 1 else ""))

    def on_log(self, _):
        self.st["logx"], self.st["logy"] = (self.lx.isChecked(),
                                            self.ly.isChecked())
        self.redraw()

    def to_clipboard(self):
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())
        self.win.log("Info: Chart saved to clipboard.")

    def on_verify(self):
        """
        A cursor reading should be checkable against the simulator:
        ngspice is run once on the exact device under the cursor and the table is
        shown against it. It is the ONE place the window starts a
        simulation, so it happens only on this explicit click, and the
        netlist it runs is left on disk (cursor_exact.spice) to re-run
        by hand -- the test bench stays yours.

        The operating point is simulated at the table's REFERENCE finger
        width, where every width-independent quantity (gm/ID, gm/gds,
        fT, VA, ...) and every per-width density is exact; the sized W
        is separate, exact arithmetic. So the comparison is of the bias
        point the chart drew, not of the rounding to whole fingers.
        """
        if not self.members or self.st["cursor"] is None:
            self.win.log(f"Plot {self.name}: put the cursor on the plot "
                         f"first (click it), then Verify.")
            return
        m = self.member()
        pv = self.pvalue()
        d = self.devices_at(self.st["cursor"],
                            [pv if pv is not None else 0.0])
        if not np.isfinite(d["VGS"][0]):
            self.win.log(f"Plot {self.name}: no device under the cursor "
                         f"to verify.")
            return
        L, VGS, VDS, VSB = (float(d[k][0])
                            for k in ("L", "VGS", "VDS", "VSB"))
        self.win.log(f"Plot {self.name}: running ngspice at {m['dev']} "
                     f"L={eng(L)} VGS={eng(VGS)} VDS={eng(VDS)} "
                     f"VSB={eng(VSB)} ...")
        QtWidgets.QApplication.processEvents()
        try:
            from gmid_cursor import Cursor
            cur = Cursor(m["sa"].dir)
            spice, why = cur.spice_at(L, VGS, VDS, VSB)
        except Exception as exc:                             # noqa: BLE001
            self.win.log(f"Plot {self.name}: verify failed: {exc}")
            return
        if spice is None:
            self.win.log(f"Plot {self.name}: ngspice did not verify: "
                         f"{why}")
            QtWidgets.QMessageBox.warning(
                self, "Verify", f"ngspice could not verify this point:\n\n"
                f"{why}")
            return
        self.show_verify(cur, m, L, VGS, VDS, VSB, spice)

    def show_verify(self, cur, m, L, VGS, VDS, VSB, spice):
        rows = []
        worst = (0.0, "")
        for _title, items in cur.card(L, VGS, VDS, VSB, spice):
            for nm, v, unit, sv in items:
                if sv is None:
                    continue
                e = (v / sv - 1.0) * 100.0 if sv != 0 else float("nan")
                if np.isfinite(e) and abs(e) > worst[0]:
                    worst = (abs(e), nm)
                rows.append((nm, v, sv, e, unit))
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Verify  --  Plot {self.name}  ({m['dev']})")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            f"ngspice at L={eng(L)}  VGS={eng(VGS)}  VDS={eng(VDS)}  "
            f"VSB={eng(VSB)}  (T={eng(cur.lut.temp_c)}C)\n"
            f"operating point at the table's reference width; W scales "
            f"exactly."))
        t = QtWidgets.QTableWidget(len(rows), 4)
        t.setHorizontalHeaderLabels(["", "table", "ngspice", "table off"])
        t.verticalHeader().setVisible(False)
        for r, (nm, val, sv, e, unit) in enumerate(rows):
            for col, txt in enumerate((f"{nm} [{unit}]" if unit else nm,
                                       eng(val), eng(sv),
                                       f"{e:+.3f}%" if np.isfinite(e)
                                       else "--")):
                it = QtWidgets.QTableWidgetItem(txt)
                if col == 3 and np.isfinite(e) and abs(e) > 1.0:
                    it.setForeground(QtGui.QColor("#b02020"))
                t.setItem(r, col, it)
        t.resizeColumnsToContents()
        t.setMinimumWidth(430)
        t.setMinimumHeight(min(26 * len(rows) + 30, 560))
        v.addWidget(t)
        lab = QtWidgets.QLabel(
            f"worst table error on this card: {worst[0]:.3g} % on "
            f"{worst[1]}.   netlist: cursor_exact.spice")
        lab.setStyleSheet("color:#444;")
        v.addWidget(lab)
        bb = QtWidgets.QDialogButtonBox(
            _enum(QtWidgets.QDialogButtonBox, "StandardButton", "Close"))
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        self.win.log(f"Plot {self.name}: verified -- worst table error "
                     f"{worst[0]:.3g}% on {worst[1]}")
        dlg.exec()

    # ---- state ----------------------------------------------------------
    def load(self, st, quiet=False):
        self.st = dict(blank_state(), **st)
        self.st["traces"] = list(self.st["traces"])
        self.lx.blockSignals(True)
        self.ly.blockSignals(True)
        self.lx.setChecked(self.st["logx"])
        self.ly.setChecked(self.st["logy"])
        self.lx.blockSignals(False)
        self.ly.blockSignals(False)
        self.members, self.xs, self.ys = [], None, []
        if self.st["traces"]:
            try:
                self.compute()
            except ValueError as exc:
                if not quiet:
                    self.win.log(f"Plot {self.name}: {exc}")
        self.redraw()


class FigurePage(QtWidgets.QWidget):
    def __init__(self, win, number):
        super().__init__()
        self.number = number
        g = QtWidgets.QGridLayout(self)
        self.cells = []
        for k, s in enumerate(SLOTS):
            c = PlotCell(win, f"{number}{s}")
            g.addWidget(c, k // 2, k % 2)
            self.cells.append(c)


class Window(QtWidgets.QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.eng = Engine(root)
        self.setWindowTitle(f"SG13G2 gm/ID Toolbox  --  Sizing Assistant   [{root}]")
        self.resize(1500, 900)
        self.active = None
        self._filling = False

        # ---- centre: figure tabs over the log --------------------------
        self.tabs = QtWidgets.QTabWidget()
        add = QtWidgets.QToolButton()
        add.setText("+")
        add.clicked.connect(self.add_figure)
        self.tabs.setCornerWidget(add)
        self.tabs.currentChanged.connect(self.on_tab)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumHeight(110)
        split = QtWidgets.QSplitter(_enum(QtCore.Qt, "Orientation",
                                          "Vertical"))
        split.addWidget(self.tabs)
        split.addWidget(self.logbox)
        split.setStretchFactor(0, 1)

        # ---- right panel ------------------------------------------------
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(350)
        pv = QtWidgets.QVBoxLayout(panel)
        row = QtWidgets.QHBoxLayout()
        self.slot_btn = []
        for k, s in enumerate(SLOTS):
            b = QtWidgets.QPushButton(f"Plot {s}")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=k: self.select(
                self.page().cells[i]))
            row.addWidget(b)
            self.slot_btn.append(b)
        pv.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Import:"))
        self.imp = QtWidgets.QComboBox()
        row.addWidget(self.imp, 1)
        ok = QtWidgets.QPushButton("OK")
        ok.clicked.connect(self.do_import)
        row.addWidget(ok)
        pv.addLayout(row)

        box = QtWidgets.QGroupBox("LUT Settings")
        form = QtWidgets.QGridLayout(box)
        self.dev = QtWidgets.QComboBox()
        self.dev.addItems(sorted(self.eng.tables))
        self.corner = QtWidgets.QComboBox()
        self.temp = QtWidgets.QComboBox()
        self.all_c = QtWidgets.QCheckBox("All")
        self.all_t = QtWidgets.QCheckBox("All")
        self.cmp_btn = QtWidgets.QToolButton()
        self.cmp_btn.setText("Compare…")
        self.cmp_btn.setToolTip("overlay several devices on one plot "
                                "(Device Comparison)")
        self.cmp_btn.clicked.connect(self.pick_compare)
        form.addWidget(QtWidgets.QLabel("LUT"), 0, 0)
        form.addWidget(self.dev, 0, 1)
        form.addWidget(self.cmp_btn, 0, 2)
        form.addWidget(QtWidgets.QLabel("Corner"), 1, 0)
        form.addWidget(self.corner, 1, 1)
        form.addWidget(self.all_c, 1, 2)
        form.addWidget(QtWidgets.QLabel("Temp (°C)"), 2, 0)
        form.addWidget(self.temp, 2, 1)
        form.addWidget(self.all_t, 2, 2)
        pv.addWidget(box)

        form = QtWidgets.QGridLayout()
        self.size_kind = QtWidgets.QComboBox()
        self.size_kind.addItems(SIZE_KINDS)
        self.inv_kind = QtWidgets.QComboBox()
        self.inv_kind.addItems(INV_KINDS)
        self.e_size, self.e_inv = (QtWidgets.QLineEdit(),
                                   QtWidgets.QLineEdit())
        self.e_L, self.e_VDS, self.e_VSB = (QtWidgets.QLineEdit(),
                                            QtWidgets.QLineEdit(),
                                            QtWidgets.QLineEdit())
        for r, (lab, ed) in enumerate([
                (self.size_kind, self.e_size), (self.inv_kind, self.e_inv),
                (QtWidgets.QLabel("L"), self.e_L),
                (QtWidgets.QLabel("VDS"), self.e_VDS),
                (QtWidgets.QLabel("VSB"), self.e_VSB)]):
            form.addWidget(lab, r, 0)
            form.addWidget(ed, r, 1)
            ed.setPlaceholderText("empty = sweep")
            ed.editingFinished.connect(self.on_fields)
        pv.addLayout(form)
        for w in (self.dev, self.corner, self.temp, self.size_kind,
                  self.inv_kind):
            w.activated.connect(self.on_fields)
        self.dev.activated.connect(self.fill_corner_temp)
        self.dev.activated.connect(self.on_dev_pick)
        self.all_c.toggled.connect(self.on_fields)
        self.all_t.toggled.connect(self.on_fields)

        pv.addWidget(QtWidgets.QLabel("Results:"))
        self.table = QtWidgets.QTableWidget(len(RESULTS), 3)
        self.table.setHorizontalHeaderLabels(["Name", "Value", ""])
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 60)
        self.table.setEditTriggers(_enum(QtWidgets.QAbstractItemView,
                                         "EditTrigger", "NoEditTriggers"))
        for r, (n, _e, _u) in enumerate(RESULTS):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(n))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(""))
            b = QtWidgets.QToolButton()
            b.setText("Plot")
            b.setPopupMode(_enum(QtWidgets.QToolButton,
                                 "ToolButtonPopupMode", "MenuButtonPopup"))
            menu = QtWidgets.QMenu(b)
            menu.addAction("Add to this plot").triggered.connect(
                lambda _=False, nm=n: self.plot(nm, add=True))
            b.setMenu(menu)
            b.clicked.connect(lambda _=False, nm=n: self.plot(nm))
            self.table.setCellWidget(r, 2, b)
        pv.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Y-Expr"))
        self.yexpr = QtWidgets.QLineEdit()
        self.yexpr.setPlaceholderText("gm/ID*fT    or    Vov = VGS - VTH")
        self.yexpr.returnPressed.connect(lambda: self.plot(
            self.yexpr.text()))
        row.addWidget(self.yexpr, 1)
        pv.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Plot")
        b.clicked.connect(lambda: self.plot(self.yexpr.text()))
        row.addWidget(b)
        b = QtWidgets.QPushButton("Add")
        b.clicked.connect(lambda: self.plot(self.yexpr.text(), add=True))
        row.addWidget(b)
        b = QtWidgets.QPushButton("Clear plot")
        b.clicked.connect(self.clear_plot)
        row.addWidget(b)
        pv.addLayout(row)

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.addWidget(split, 1)
        h.addWidget(panel)
        self.setCentralWidget(central)

        m = self.menuBar().addMenu("File")
        for text, fn in (("Save State...", self.save_state),
                         ("Load State...", self.load_state),
                         ("Export plot data (CSV)...", self.export_csv),
                         ("Save plot picture...", self.save_png),
                         ("Quit", self.close)):
            m.addAction(text).triggered.connect(lambda _=False, f=fn: f())

        self.add_figure()
        self.fill_corner_temp()
        n = sum(len(v) for v in self.eng.tables.values())
        self.log(f"{n} table(s) for {len(self.eng.tables)} device(s) found "
                 f"in {os.path.abspath(root)}")
        if not self.eng.tables:
            self.log("NO TABLES HERE. Start this in the directory that "
                     "holds the lut_* folders.")

    # ---- plumbing -------------------------------------------------------
    def log(self, msg):
        t = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.logbox.appendPlainText(f"[{t}] - {msg}")

    def page(self):
        return self.tabs.currentWidget()

    def add_figure(self):
        n = self.tabs.count() + 1
        pg = FigurePage(self, n)
        dev = self.dev.currentText() if hasattr(self, "dev") else ""
        for c in pg.cells:
            c.st["dev"] = dev
        self.tabs.addTab(pg, f"Figure {n}")
        self.tabs.setCurrentWidget(pg)

    def on_tab(self, _):
        pg = self.page()
        if pg is not None and hasattr(self, "table"):
            self.select(pg.cells[0])

    def all_cells(self):
        return [c for i in range(self.tabs.count())
                for c in self.tabs.widget(i).cells]

    def select(self, cell):
        if cell is self.active:
            return
        self.active = cell
        for c in self.all_cells():
            c.mark(c is cell)
        pg = self.page()
        for b, c in zip(self.slot_btn, pg.cells):
            b.setText(f"Plot {c.name}")
            b.setChecked(c is cell)
        self.imp.clear()
        self.imp.addItems([f"Plot {c.name}" for c in self.all_cells()
                           if c is not cell])
        self.show_state(cell.st)

    # ---- fields <-> state -----------------------------------------------
    def fill_corner_temp(self, *_):
        dev = self.dev.currentText()
        for box, items in ((self.corner, self.eng.corners(dev)),
                           (self.temp, [f"{t:g}" for t in
                                        self.eng.temps(dev)])):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(items)
            pick = cur if cur in items else (
                "TT" if "TT" in items else "27" if "27" in items
                else (items[0] if items else ""))
            box.setCurrentText(pick)
            box.blockSignals(False)

    def on_dev_pick(self, *_):
        # picking ONE device from the dropdown ends a comparison
        if self.active is not None:
            self.active.st["devs"] = []
        self.cmp_btn.setText("Compare…")

    def pick_compare(self):
        """choose the set of devices to overlay (Device Comparison)"""
        if self.active is None:
            return
        st = self.active.st
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Device Comparison")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            "Overlay these devices on the plot, same settings on each:"))
        boxes = []
        chosen = set(st.get("devs") or [st["dev"]])
        for d in sorted(self.eng.tables):
            cb = QtWidgets.QCheckBox(d)
            cb.setChecked(d in chosen)
            v.addWidget(cb)
            boxes.append(cb)
        bb = QtWidgets.QDialogButtonBox(
            _enum(QtWidgets.QDialogButtonBox, "StandardButton", "Ok")
            | _enum(QtWidgets.QDialogButtonBox, "StandardButton", "Cancel"))
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if not dlg.exec():
            return
        picked = [cb.text() for cb in boxes if cb.isChecked()]
        if not picked:
            return
        st["devs"] = picked if len(picked) > 1 else []
        if len(picked) == 1:
            st["dev"] = picked[0]
        st["pval"] = None
        self.cmp_btn.setText(f"Compare ({len(picked)})" if len(picked) > 1
                             else "Compare…")
        self.active.load(st)
        self.show_state(st)

    def show_state(self, st):
        self._filling = True
        n = len(st.get("devs") or [])
        self.cmp_btn.setText(f"Compare ({n})" if n > 1 else "Compare…")
        if st["dev"] and self.dev.findText(st["dev"]) >= 0:
            self.dev.setCurrentText(st["dev"])
        self.fill_corner_temp()
        self.corner.setCurrentText(st["corner"])
        self.temp.setCurrentText(f"{st['temp']:g}")
        self.all_c.setChecked(st["all_corner"])
        self.all_t.setChecked(st["all_temp"])
        self.size_kind.setCurrentText(st["size_kind"])
        self.inv_kind.setCurrentText(st["inv_kind"])
        for ed, k in ((self.e_size, "size"), (self.e_inv, "inv"),
                      (self.e_L, "L"), (self.e_VDS, "VDS"),
                      (self.e_VSB, "VSB")):
            ed.setText(st[k])
        self._filling = False
        self.refresh_results()

    def read_state(self, st):
        st["dev"] = self.dev.currentText()
        st["corner"] = self.corner.currentText() or "TT"
        try:
            st["temp"] = float(self.temp.currentText())
        except ValueError:
            st["temp"] = 27.0
        st["all_corner"] = self.all_c.isChecked()
        st["all_temp"] = self.all_t.isChecked()
        st["size_kind"] = self.size_kind.currentText()
        st["inv_kind"] = self.inv_kind.currentText()
        for ed, k in ((self.e_size, "size"), (self.e_inv, "inv"),
                      (self.e_L, "L"), (self.e_VDS, "VDS"),
                      (self.e_VSB, "VSB")):
            st[k] = ed.text().strip()

    def on_fields(self, *_):
        if self._filling or self.active is None:
            return
        self.read_state(self.active.st)
        self.refresh_results()
        # A changed setting must change the CURVE, not only the Results
        # column. Without this, picking another corner or temperature
        # updated the numbers while the old curve stayed on screen, so
        # the plot silently disagreed with the panel that produced it --
        # the worst kind of wrong, because it looks right.
        cell = self.active
        if cell.st["traces"]:
            try:
                cell.compute()
            except Exception as exc:                         # noqa: BLE001
                self.log(f"Plot {cell.name}: {exc}")
                cell.members, cell.xs, cell.ys = [], None, []
            cell.redraw()

    def refresh_results(self):
        """numbers in the Value column when the five fields name ONE device"""
        vals = [""] * len(RESULTS)
        st = self.active.st if self.active else None
        if st and st["dev"]:
            try:
                members, which = self.eng.families(st)
                if which is None and len(members) == 1:
                    m = members[0]
                    d = m["sa"].device(m["size"], m["inv"], m["f"]["L"],
                                       m["f"]["VDS"], m["f"]["VSB"])
                    if d is None:
                        vals = ["no device"] * len(RESULTS)
                    else:
                        vals = []
                        for n, _e, u in RESULTS:
                            v = m["sa"].value(n, d)
                            vals.append(eng(v) + (" " + u if u else "")
                                        if np.isfinite(v) else "--")
            except Exception:                                # noqa: BLE001
                pass
        for r, v in enumerate(vals):
            self.table.item(r, 1).setText(v)

    # ---- actions --------------------------------------------------------
    def plot(self, expr, add=False):
        expr = expr.strip()
        if not expr or self.active is None:
            return
        cell = self.active
        self.read_state(cell.st)
        old = list(cell.st["traces"])
        cell.st["traces"] = (old + [expr]) if add else [expr]
        try:
            cell.compute()
        except Exception as exc:                             # noqa: BLE001
            cell.st["traces"] = old
            self.log(f"Plot {cell.name}: {exc}")
            return
        if cell.st["cursor"] is None:
            cell.st["cursor"] = float(cell.xs[len(cell.xs) // 2])
        cell.redraw()
        self.log(f"Plot {cell.name}: {', '.join(cell.st['traces'])} "
                 f"versus {cell.which}, {len(cell.members)} curve set(s)")

    def clear_plot(self):
        if self.active is not None:
            self.active.st["traces"] = []
            self.active.load(self.active.st, quiet=True)

    def do_import(self):
        src = [c for c in self.all_cells()
               if f"Plot {c.name}" == self.imp.currentText()]
        if not src or self.active is None:
            return
        st = json.loads(json.dumps(src[0].st))
        st["traces"], st["cursor"] = list(self.active.st["traces"]), None
        self.active.load(st)
        self.show_state(self.active.st)
        self.log(f"Plot {self.active.name}: settings imported from "
                 f"Plot {src[0].name}")

    def save_state(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save State", "state.json", "State (*.json)")
        if not path:
            return
        if self.active is not None:
            self.read_state(self.active.st)
        with open(path, "w") as fh:
            json.dump(dict(figures=[[c.st for c in self.tabs.widget(i).cells]
                                    for i in range(self.tabs.count())]),
                      fh, indent=1)
        self.log(f"State saved to {path}")

    def load_state(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load State", "", "State (*.json)")
        if not path:
            return
        with open(path) as fh:
            figs = json.load(fh)["figures"]
        while self.tabs.count() < len(figs):
            self.add_figure()
        for i, cells in enumerate(figs):
            for c, st in zip(self.tabs.widget(i).cells, cells):
                c.load(st)
        self.tabs.setCurrentIndex(0)
        self.active = None
        self.select(self.page().cells[0])
        self.log(f"State loaded from {path}")

    def export_csv(self, path=None):
        c = self.active
        if c is None or c.xs is None:
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export", f"plot_{c.name}.csv", "CSV (*.csv)")
        if not path:
            return
        head, cols = [c.which], [c.xs]
        for m, row in zip(c.members, c.ys):
            for t, y in zip(c.st["traces"], row):
                head.append(f"{split_expr(t)[0]} [{m['label']}]")
                cols.append(y)
        with open(path, "w") as fh:
            fh.write(",".join(head) + "\n")
            for r in np.column_stack(cols):
                fh.write(",".join("" if not np.isfinite(v) else f"{v:.8g}"
                                  for v in r) + "\n")
        self.log(f"Plot {c.name}: data written to {path}")

    def save_png(self, path=None):
        c = self.active
        if c is None:
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save picture", f"plot_{c.name}.png", "PNG (*.png)")
        if path:
            c.fig.savefig(path, dpi=150)
            self.log(f"Plot {c.name}: picture written to {path}")


def selftest(root, shot):
    """drive the window without a person: four plots, every cursor mode"""
    app = QtWidgets.QApplication(sys.argv[:1])
    w = Window(root)
    w.show()
    dev = "sg13_hv_nmos" if "sg13_hv_nmos" in w.eng.tables \
        else sorted(w.eng.tables)[0]
    cells = w.page().cells

    def setup(cell, **kw):
        w.select(cell)
        st = dict(blank_state(dev), **kw)
        cell.load(st, quiet=True)
        w.show_state(cell.st)

    setup(cells[0], size_kind="ID", size="10u", inv_kind="Vstar",
          inv="200m", L="", VDS="1.5", VSB="0")
    w.plot("ro")
    cells[0].set_cursor(2e-6)
    setup(cells[1], size_kind="W", size="40u", inv_kind="VGS", inv="",
          L="2u", VDS="1.5", VSB="0")
    w.plot("Vstar")
    w.plot("Vov = VGS - VTH", add=True)
    cells[1].set_cursor(0.817)
    setup(cells[2], size_kind="ID", size="10u", inv_kind="Vstar",
          inv="200m", L="0.5u,1u,2u,5u", VDS="", VSB="0")
    w.plot("gm/gds")
    cells[2].on_param(2)
    cells[2].set_cursor(1.5)
    setup(cells[3], size_kind="ID", size="80u", inv_kind="gm/ID",
          inv="", L="0.5u, 1u:1u:3u", VDS="0.9", VSB="0")
    w.plot("gm/gds")
    cells[3].set_cursor(15.71)
    cells[3].pbox.setEditText("0.7u")
    cells[3].on_param_typed()
    cells[3].yout.setText("150")
    cells[3].on_ytyped()
    print("off-trace D:", cells[3].xedit.text(), cells[3].yout.text(),
          cells[3].pbox.currentText())
    w.select(cells[0])
    w.e_L.setText("2u")
    w.on_fields()
    w.save_state(os.path.join(os.path.dirname(shot) or ".", "state.json"))
    app.processEvents()
    w.grab().save(shot)
    print(w.logbox.toPlainText())
    print("results:", [(w.table.item(r, 0).text(), w.table.item(r, 1).text())
                       for r in (0, 1, 2, 7, 17)])
    print("readouts A:", cells[0].xedit.text(), cells[0].yout.text())
    return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else "."
    if "--selftest" in sys.argv:
        return selftest(root, args[1] if len(args) > 1 else "selftest.png")
    app = QtWidgets.QApplication(sys.argv[:1])
    w = Window(root)
    w.show()
    return getattr(app, "exec", None)() if hasattr(app, "exec") \
        else app.exec_()


if __name__ == "__main__":
    sys.exit(main())
