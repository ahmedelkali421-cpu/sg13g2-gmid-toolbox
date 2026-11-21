# sg13g2-gmid-toolbox

An open-source **gm/ID sizing toolbox** for the [IHP SG13G2](https://github.com/IHP-GmbH/IHP-Open-PDK)
open PDK, built on precomputed lookup tables (LUTs) extracted with
ngspice.

You give it an operating point — a current, an inversion level, a length —
and it gives you back a transistor: width, gate voltage, and every
small-signal, capacitance and noise parameter at that point. No quadratic
model, no guesswork: every number is interpolated from data the simulator
produced, and the interpolation error is *measured*, not assumed.

![screenshot](screenshot.png)

---

## Why

The square-law equations taught in textbooks do not describe a modern
short-channel device. The industry answer is the **gm/ID methodology**:
characterise the device once, over a grid, and design from the stored
data instead of from an equation.

This toolbox is a complete implementation of that idea for SG13G2:
table generation, a 4-D interpolation engine, a sizing assistant, design
charts, and a cursor that reads any point *between* the curves.

## Accuracy

This is the part that matters, and it is measured rather than claimed.
Every table is verified against ngspice at points that deliberately fall
**between** grid nodes, where interpolation error is largest:

| device | median residual | worst case |
|---|---|---|
| LV NMOS | 0.0001 % | 0.065 % |
| LV PMOS | 0.0000 % | 0.004 % |
| HV NMOS | 0.0004 % | 0.426 % |
| HV PMOS | 0.0002 % | 0.289 % |

The worst cases sit in the short-channel region, where the model itself
is sharpest. The tool **tells you the error at the point you are
looking at**, and a `Verify` button runs ngspice on that exact device so
you can see table vs simulator side by side.

## What you get

| | |
|---|---|
| devices | `sg13_lv_nmos`, `sg13_lv_pmos`, `sg13_hv_nmos`, `sg13_hv_pmos` |
| corners | TT, SS, FF |
| temperatures | −40, 0, 27, 85, 125 °C |
| stored per point | 33 model parameters (ID, gm, gmb, gds, all caps, noise, …) |
| derived | gm/ID, V\*, fT, gm/gds, VA, VDSAT, γ, fknee, Ron, … |

Also included: finger-count corrections (overlap and junction
capacitances scale with the number of fingers, and the tool models
that), mismatch/Pelgrom extraction, and corner comparison.

---

## Install

```bash
git clone https://github.com/<you>/sg13g2-gmid-toolbox
cd sg13g2-gmid-toolbox
pip install -r requirements.txt
```

You also need **ngspice** and the **SG13G2 PDK**. The easiest route is
[IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS), which ships
both. Point the toolbox at your models:

```bash
export SG13G2_MODELS=/foss/pdks/ihp-sg13g2/libs.tech/ngspice/models
```

(That default is already the IIC-OSIC-TOOLS path, so inside the
container it works out of the box.)

## Quickstart

### 1. Build a table

One device, one corner, one temperature. Takes roughly 20 minutes and
produces a ~0.5 GB table.

```bash
python3 make_sweep.py hv tt                 # writes the netlist
ngspice -b sweep_hv_mos_tt.spice            # runs the sweep
python3 pack_lut.py lut_hv_tt.raw --flavour hv
rm lut_hv_tt.raw                            # the raw is large and no longer needed
```

Repeat with `make_edge.py` for the companion single-finger table, which
the finger-count correction needs.

### 2. Size a transistor from the command line

```bash
python3 gmid_sa.py lut_hv_tt --ID 10u --VSTAR 200m --L 2u --VDS 1.5 --VSB 0
```

```
  ID                10u  A
  W              14.64u  m
  VGS              746m  V
  gm/ID              10  S/A
  fT           187.6MEG  Hz
  gm/gds          224.7
  ro           2.247MEG  ohm
  ...
```

### 3. Open the window

```bash
python3 gmid_gui.py
```

## The one idea to understand

A transistor's operating point is **five fields**:

| field | alternatives |
|---|---|
| size | `W` or `ID` |
| inversion level | `VGS` or `V*` or `gm/ID` |
| length | `L` |
| drain bias | `VDS` |
| body bias | `VSB` |

**Fill four and leave one empty — the empty one becomes the sweep axis.**
Fill all five and you get one sized device instead of a curve.

That single rule drives the whole tool. Leave `VGS` empty and you get
curves against VGS; leave `L` empty and you get curves against length at
a fixed current and inversion level.

## Tools

| script | what it does |
|---|---|
| `gmid_gui.py` | the window: five fields, four plots, results table, cursor |
| `gmid_sa.py` | the same engine from the command line |
| `gmid_cursor.py` | point lookup — give a chart point, get the device |
| `gmid_xplore.py` | design charts (the canonical nine-panel sheet) |
| `gmid_size.py` | size from specifications |
| `gmid_corner.py` | compare corners |
| `gmid_mm.py` | mismatch / Pelgrom |
| `make_sweep.py`, `make_edge.py`, `pack_lut.py` | table generation |
| `make_verify.py`, `check_verify.py` | accuracy verification against ngspice |

### The cursor

Click anywhere on a chart — including the white space *between* the
drawn curves — and the tool solves for the device at that point and
reads every parameter there. Type a target value and it solves backwards
for the length that gives it. Press `Verify` and ngspice is run once on
that exact device, with the table and the simulator shown side by side.

---

## Status and limits

This is working software that has been used for real sizing, but it is
young and has had exactly one user. **Bug reports are the most useful
thing you can send.**

Known limits:

- Tables are not included (a full set is >100 GB). You generate your own;
  see the quickstart.
- SG13G2 only so far. The device definitions live in one table at the top
  of `make_sweep.py`; another PDK means adding an entry there, not
  rewriting the tool.
- Transistor level only. There is no circuit-level topology library or
  optimiser.
- The window has been tested on PyQt5 under Linux.

## Credits

The gm/ID methodology and the lookup-table approach are not mine. This
implementation follows the published work of **Prof. Hesham Omran**
(Ain Shams University), in particular:

- H. Omran et al., *"Analog IC Design Using Precomputed Lookup Tables:
  Challenges and Solutions"*, IEEE Access, 2020.
- H. Omran, *"Toward Memory-Efficient Analog Design Using Precomputed
  Lookup Tables"*, Electronics 13(18):3776, 2024.

and the classic treatment in Jespers & Murmann, *Systematic Design of
Analog CMOS Circuits*.

This project is an **independent open-source implementation** written
from the published methodology.

Thanks to the [IHP](https://github.com/IHP-GmbH/IHP-Open-PDK) team for
the open PDK and to [IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS)
for the toolchain.

## License

MIT — see [LICENSE](LICENSE).
