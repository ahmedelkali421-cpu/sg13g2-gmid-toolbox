"""
Streaming reader for ngspice binary rawfiles that contain MANY plots
(what `set appendwrite` produces).

The whole point is that it never holds more than one plot in memory, so a
2.3 GB file with 1221 plots is read in a few hundred MB of RAM.

Rawfile layout, repeated once per plot:

    Title: ...
    Date: ...
    Plotname: ...
    Flags: real
    No. Variables: N
    No. Points: M
    Variables:
    <tab>0<tab>name<tab>type
    ...
    Binary:
    <M*N IEEE-754 float64, point-major>
"""

import numpy as np


def _readline(fh):
    line = fh.readline()
    if not line:
        return None
    return line.decode("latin-1", errors="replace").rstrip("\r\n")


def iter_plots(path, want=None):
    """
    Yield (index, header_dict, {varname: 1-D float64 array}) per plot.

    want : optional set of variable names to keep. Everything else is
           dropped right after the read, which keeps memory flat.
    """
    with open(path, "rb") as fh:
        idx = 0
        while True:
            line = _readline(fh)
            while line is not None and line.strip() == "":
                line = _readline(fh)
            if line is None:
                return

            hdr, names = {}, []
            while True:
                if line is None:
                    return
                low = line.lower()
                if low.startswith("variables:"):
                    for _ in range(int(hdr["no. variables"])):
                        parts = _readline(fh).split()
                        names.append(parts[1])
                    line = _readline(fh)
                    continue
                if low.startswith("binary:"):
                    break
                if low.startswith("values:"):
                    raise IOError(
                        f"{path}: plot {idx} is ASCII. This reader is binary "
                        f"only. Re-run with  set filetype = binary")
                if ":" in line:
                    k, v = line.split(":", 1)
                    hdr[k.strip().lower()] = v.strip()
                line = _readline(fh)

            npts = int(hdr["no. points"])
            nvar = int(hdr["no. variables"])
            if "complex" in hdr.get("flags", "real").lower():
                raise IOError(f"{path}: plot {idx} is complex, not expected")

            buf = np.fromfile(fh, dtype=np.float64, count=npts * nvar)
            if buf.size != npts * nvar:
                raise IOError(
                    f"{path}: plot {idx} truncated "
                    f"({buf.size} of {npts*nvar} values). The sweep probably "
                    f"did not finish.")
            mat = buf.reshape(npts, nvar)

            if want is None:
                data = {n: mat[:, i].copy() for i, n in enumerate(names)}
            else:
                data = {n: mat[:, i].copy()
                        for i, n in enumerate(names) if n in want}
            del buf, mat

            yield idx, hdr, data
            idx += 1


def short_name(v):
    """
    Reduce an ngspice vector name to the bare PSP parameter.

    ngspice wraps saved device parameters according to the unit it infers
    from the OSDI model, so the same sweep produces three spellings:

        i(@n.xm1.nsg13_hv_nmos[ids])   ->  ids
        v(@n.xm1.nsg13_hv_nmos[vth])   ->  vth
        cgg                            ->  cgg
        v(v-sweep)                     ->  v-sweep
    """
    s = v.strip()
    if len(s) > 2 and s[0] in "viVI" and s[1] == "(" and s.endswith(")"):
        s = s[2:-1]
    if "[" in s and s.endswith("]"):
        return s[s.index("[") + 1:-1]
    return s
