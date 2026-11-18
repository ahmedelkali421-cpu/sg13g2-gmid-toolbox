import glob, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "xplore"


def read(path):
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split(",")
    M = np.genfromtxt(path, delimiter=",", skip_header=1)
    M = np.atleast_2d(M)
    return head, M


def look(pat, title, unit=""):
    f = sorted(glob.glob(os.path.join(d, pat)))
    if not f:
        print(f"  {title}: no CSV matching {pat} in {d}/")
        return
    head, M = read(f[0])
    xcol = [j for j, h in enumerate(head) if h.split("@")[0].upper() == "GM/ID"]
    ycol = [j for j in range(len(head)) if j not in xcol]
    print(f"\n{title}   [{os.path.basename(f[0])}]")
    print(f"  {'L':>9} {'at gm/ID=5':>13} {'=10':>13} {'=15':>13} "
          f"{'=20':>13} {'=25':>13}   {'span':>11}")
    for j in ycol:
        L = head[j].split("@")[-1]
        x, y = M[:, j - 1], M[:, j]
        row = []
        for t in (5, 10, 15, 20, 25):
            k = int(np.nanargmin(np.abs(x - t)))
            row.append(y[k])
        v = y[np.isfinite(y)]
        # decades, not per cent: a quantity that moves four orders of
        # magnitude across the chart reports a span of 27315 %, which is
        # a true number that tells you nothing
        sp = np.log10(v.max() / v.min()) if v.size and v.min() > 0 else 0.0
        print(f"  {L:>9} " + " ".join(f"{r:13.5g}" for r in row)
              + f"   {sp:7.2f} dec")


look("SID*", "GAMMA = SID / (4 k T gm)")
look("FKNEE*", "FLICKER CORNER [Hz]")
