#!/usr/bin/env python3
"""
=============================================================================
SG13G2 gm/ID Toolbox | DEMO TABLE PACK
=============================================================================
    python3 make_demo_lut.py SRC DST                       one pack, all devices
    python3 make_demo_lut.py SRC DST --pack gmid-demo-tables.tar.xz
        [--every 3] [--dense 2.5] [--check 400]

SRC is the folder holding the full tables (lut_* and edge_* side by side),
DST a new, empty folder. For every device found in SRC, the table at the
nominal corner (TT) and 27 C is shrunk, together with its edge table,
checked against the full table, and written to DST under one uniform name:

    DST/lut_<device>_tt_27      DST/edge_<device>_tt_27
    e.g. lut_hv_nmos_tt_27, edge_lv_pmos_tt_27

A full table is 0.9 - 1.6 GB. Nobody downloads 5 GB to find out whether
a tool runs. So the demo keeps EVERY stored length up to dense x Lmin
(default 2.5, where short-channel effects make L matter), only every Nth
length above that (default 3, the longest always kept), and drops the
leakage terms no chart reads.

NO SIMULATION. It only rewrites what is already packed: the demo is
EXACT at the lengths it keeps.

WHAT IT COSTS IS MEASURED, NOT ASSUMED. After shrinking, both tables are
evaluated at random operating points whose L falls between the kept
lengths -- the worst case, where the dropped lengths were -- and the error
of the demo against the full table is written into the demo's own
meta.json ("demo_check"). The number travels with the data.

THE EDGE TABLE MOVES WITH IT. The finger-count correction pairs the main
table with an ng = 1 table on the SAME L axis, and refuses a pair whose
axes differ. So the edge table is cut at the same lengths; shrinking the
main table alone would silently switch the correction off.

Side files: mismatch.json and finger_laws.json do not depend on the L
grid and are copied. l_accuracy.json belongs to the FULL grid and would
understate the demo's error, so it is deliberately left out.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

# Leakage and avalanche terms. No chart reads them; everything the
# Results table and the finger-count correction need is kept.
DROP = ["igs", "igd", "igb", "iavl", "ijd", "cigid"]
COPY = ["mismatch.json", "finger_laws.json"]
CHECK = ["ID", "GM/ID", "GM/GDS", "FT", "CGG"]


def shrink(src, dst, keep, drop=True):
    """cut one table (main or edge) to the stored lengths in keep"""
    with open(os.path.join(src, "meta.json")) as fh:
        m = json.load(fh)
    out_params = [p for p in m["params"] if not (drop and p in DROP)]
    os.makedirs(dst, exist_ok=True)
    for p in out_params:
        f = os.path.join(src, p + ".npy")
        if not os.path.exists(f):
            continue
        a = np.load(f, mmap_mode="r")
        np.save(os.path.join(dst, p + ".npy"), np.ascontiguousarray(a[keep]))
    nL = len(m["L"])
    m["L"] = [m["L"][i] for i in keep]
    m["leff"] = [m["leff"][i] for i in keep]
    m["params"] = out_params
    m["demo"] = True
    m["note"] = (m.get("note", "") +
                 f"  DEMO TABLE: {len(keep)} of {nL} stored lengths kept, "
                 f"leakage parameters dropped. Interpolation between "
                 f"lengths is coarser than the full table's; demo_check "
                 f"in this file says by how much.").strip()
    with open(os.path.join(dst, "meta.json"), "w") as fh:
        json.dump(m, fh)
    for extra in COPY:
        s = os.path.join(src, extra)
        if os.path.exists(s):
            shutil.copy2(s, dst)
    return m


def check(full_dir, demo_dir, n=400, seed=7):
    """demo against full at random points BETWEEN the kept lengths"""
    from gmid_size import Sizer
    full = Sizer(full_dir).lut
    demo = Sizer(demo_dir).lut
    rng = np.random.default_rng(seed)
    Lk = np.asarray(demo.L)
    # one random L inside each gap of the kept axis, cycled
    gaps = [(Lk[i], Lk[i + 1]) for i in range(len(Lk) - 1)]
    vmax = float(np.max(full.VGS))
    L, VGS, VDS, VSB = [], [], [], []
    while len(L) < n:
        a, b = gaps[len(L) % len(gaps)]
        L.append(float(np.exp(rng.uniform(np.log(a), np.log(b)))))
        vsb = float(rng.uniform(0.0, 0.3 * vmax))
        VSB.append(vsb)
        VGS.append(float(rng.uniform(0.25 * vmax, vmax - vsb)))
        VDS.append(float(rng.uniform(0.15 * vmax, vmax - vsb)))
    L, VGS, VDS, VSB = map(np.asarray, (L, VGS, VDS, VSB))
    out = {"points": n, "where": "random L between kept lengths, "
           "VGS/VDS/VSB random inside the stored region", "rel_err_pct": {}}
    for q in CHECK:
        f = np.asarray(full.eval(q, L, VGS, VDS, VSB), float)
        d = np.asarray(demo.eval(q, L, VGS, VDS, VSB), float)
        ok = np.isfinite(f) & np.isfinite(d) & (np.abs(f) > 0)
        e = 100 * np.abs(d[ok] / f[ok] - 1)
        out["rel_err_pct"][q] = (dict(median=float(np.median(e)),
                                      p95=float(np.percentile(e, 95)),
                                      max=float(e.max()), n=int(ok.sum()))
                                 if e.size else None)
    return out


def short_name(device):
    """sg13_hv_nmos -> hv_nmos"""
    return device[5:] if device.startswith("sg13_") else device


def dsize(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 1
    src, dst = args[0].rstrip("/"), args[1].rstrip("/")
    opt = dict(zip(argv[::1], argv[1::1]))
    every = int(opt.get("--every", 3))
    dense = float(opt.get("--dense", 2.5))
    npts = int(opt.get("--check", 400))
    pack = opt.get("--pack")
    if os.path.exists(dst) and os.listdir(dst):
        sys.exit(f"{dst} is not empty; give a new folder")
    os.makedirs(dst, exist_ok=True)

    from gmid_sa import find_tables
    from gmid_size import edge_dir_for
    tables = find_tables(src)
    picked = []
    for dev, d in sorted(tables.items()):
        hits = [p for (c, t), p in d.items()
                if c.upper().endswith("TT") and abs(t - 27.0) < 0.5]
        if hits:
            picked.append((dev, hits[0]))
    if not picked:
        sys.exit(f"no TT / 27 C table found under {src}")

    summary = []
    for dev, main_src in picked:
        name = short_name(dev)
        main_dst = os.path.join(dst, f"lut_{name}_tt_27")
        with open(os.path.join(main_src, "meta.json")) as fh:
            Ls = json.load(fh)["L"]
        nL = len(Ls)
        # Short channels are where L matters: measured on hv_nmos with
        # every 3rd length kept, the error between kept lengths was up to
        # 8 % on ID below 0.6 um and under 0.02 % above 1.1 um. So every
        # length below dense x Lmin is kept, and only the long, smooth
        # part of the axis is thinned.
        lim = dense * Ls[0]
        short = [i for i, v in enumerate(Ls) if v <= lim * (1 + 1e-9)]
        rest = [i for i in range(nL) if i not in short]
        keep = sorted(set(short + rest[::every] + [nL - 1]))
        shrink(main_src, main_dst, keep)
        edge_src = edge_dir_for(main_src)
        if edge_src:
            shrink(edge_src, os.path.join(dst, f"edge_{name}_tt_27"), keep)
        else:
            print(f"  ! {dev}: no edge table beside {main_src}; the demo "
                  f"will run without the finger-count correction")
        res = check(main_src, main_dst, npts)
        with open(os.path.join(main_dst, "meta.json")) as fh:
            m = json.load(fh)
        m["demo_check"] = res
        with open(os.path.join(main_dst, "meta.json"), "w") as fh:
            json.dump(m, fh)
        before = dsize(main_src) + (dsize(edge_src) if edge_src else 0)
        after = dsize(main_dst) + (dsize(os.path.join(
            dst, f"edge_{name}_tt_27")) if edge_src else 0)
        summary.append((dev, nL, len(keep), before, after, res))

    print()
    for dev, nL, k, b, a, res in summary:
        print(f"  {dev}:  L {nL} -> {k}   {b/1e6:7.0f} MB -> {a/1e6:6.0f} MB")
        for q, r in res["rel_err_pct"].items():
            if r:
                print(f"      {q:7s} error vs full table: median "
                      f"{r['median']:.3f} %   95% {r['p95']:.3f} %   "
                      f"max {r['max']:.3f} %")
    tot = dsize(dst)
    print(f"\n  demo folder: {tot/1e6:.0f} MB on disk")

    if pack:
        env = dict(os.environ, XZ_OPT="-6 -T0")
        subprocess.run(["tar", "-C", dst, "-cJf", pack, "."], check=True,
                       env=env)
        h = hashlib.sha256()
        with open(pack, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        with open(pack + ".sha256", "w") as fh:
            fh.write(f"{h.hexdigest()}  {os.path.basename(pack)}\n")
        print(f"  packed: {pack}  {os.path.getsize(pack)/1e6:.0f} MB "
              f"to download\n  sha256: {h.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
