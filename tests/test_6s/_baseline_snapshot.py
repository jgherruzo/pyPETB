#           .---.        .-----------
#          /     \  __  /    ------
#         / /     \(..)/    -----
#        //////   ' \/ `   ---
#       //// / // :    : ---
#      // /   /  /`    '--
#     // /        //..\\
#   o===|========UU====UU=====-  -==========================o
#                '//||\\`
#
#   -=====================|===o  o===|======================-+
"""Regression baseline snapshot tool (developer use only, not a test).

Freezes the numerical output of the pypetb models so Phase A refactors can
be verified as behavior-preserving.

Usage:
    python _baseline_snapshot.py --save      # write reference files
    python _baseline_snapshot.py --compare   # regenerate and compare

Comparison tolerance: 1e-9 (absolute and relative).
Datasets are the same ones used by the test-suite (downloaded copies live
in ``_data/``) plus fixed-seed generated data for the Capability models.
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pypetb import Capability, Repeatability, RnR, tables  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "_data"
BASELINE = HERE / "_baseline"
TOL = 1e-9


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------
def _rnr_numeric():
    """Snapshot RnR.RnRNumeric outputs."""
    df = pd.read_csv(DATA / "RnR_Example.csv", sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    model = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    model.RnRSolve()
    out = {
        "rnr_anova.csv": model.RnRAnova(),
        "rnr_vartable.csv": model.RnR_varTable(),
        "rnr_sdtable.csv": model.RnR_SDTable(),
        "rnr_pivot.csv": model._RnRNumeric__df.reset_index(),
    }
    scalars = {
        "ndc": model.ndc,
        "dbl_Range_avg": model.dbl_Range_avg,
        "dbl_Range_UCL": model.dbl_Range_UCL,
        "dbl_Range_LCL": model.dbl_Range_LCL,
        "dbl_Avg_UCL": model.dbl_Avg_UCL,
        "dbl_Avg_LCL": model.dbl_Avg_LCL,
        "Total_avg": model.Total_avg,
        "SStechnician": model.SStechnician,
        "SSpart": model.SSpart,
        "SStotal": model.SStotal,
        "SSequipment": model.SSequipment,
        "SStechnicianxpart": model.SStechnicianxpart,
    }
    return out, scalars, model.getLog()


def _repeatability():
    """Snapshot Repeatability.RNumeric outputs."""
    df = pd.read_csv(DATA / "GRnR_perp1.csv", sep=";")
    dict_key = {"1": "Part", "2": "Measurement"}
    model = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    model.RSolve()
    out = {
        "rep_anova.csv": model.RAnova(),
        "rep_vartable.csv": model.R_varTable(),
        "rep_sdtable.csv": model.R_SDTable(),
        "rep_pivot.csv": model._RNumeric__df.reset_index(),
    }
    scalars = {
        "ndc": model.ndc,
        "dbl_Range_avg": model.dbl_Range_avg,
        "dbl_Range_UCL": model.dbl_Range_UCL,
        "dbl_Range_LCL": model.dbl_Range_LCL,
        "dbl_Avg_UCL": model.dbl_Avg_UCL,
        "dbl_Avg_LCL": model.dbl_Avg_LCL,
        "Total_avg": model.Total_avg,
        "SSpart": model.SSpart,
        "SStotal": model.SStotal,
        "SSequipment": model.SSequipment,
    }
    return out, scalars, model.getLog()


def _extract_cap_values(fig):
    """Extract the 4-value capability blocks rendered in a report figure.

    Returns a list of token lists in rendering order, e.g.
    ``[["1.5", "1.2", "1.8", "1.2"]]`` for a short term report and
    ``[[Pp, PpL, PpU, Ppk], [Cp, CpL, CpU, Cpk]]`` for a long term one.
    """
    blocks = []
    for ax in fig.axes:
        for txt in ax.texts:
            tokens = txt.get_text().split()
            if len(tokens) != 4:
                continue
            ok = True
            for token in tokens:
                if token == "*":
                    continue
                try:
                    float(token)
                except ValueError:
                    ok = False
                    break
            if ok:
                blocks.append(tokens)
    return blocks


def _capability_st():
    """Snapshot short term Capability scalars (fixed seed dataset)."""
    rng = np.random.default_rng(20240101)
    df = pd.DataFrame({"Meas": rng.normal(10.0, 0.5, 200)})
    mydict = {"value": "Meas", "batch": "", "LSL": 8.5, "HSL": 11.5,
              "goal": 10.0}
    model = Capability.Capability(df, mydict)
    fig = model.Report()
    (cp, cpl, cpu, cpk) = _extract_cap_values(fig)[0]
    scalars = {"Cp": cp, "CpL": cpl, "CpU": cpu, "Cpk": cpk}
    return {}, scalars, model.getLog()


def _capability_lt():
    """Snapshot long term Capability scalars (fixed seed dataset)."""
    rng = np.random.default_rng(20240202)
    values = []
    batches = []
    for i, offset in enumerate([0.0, 0.15, -0.1, 0.05, -0.05]):
        values.extend(rng.normal(10.0 + offset, 0.45, 40))
        batches.extend([f"B{i}"] * 40)
    df = pd.DataFrame({"Meas": values, "Lot": batches})
    mydict = {"value": "Meas", "batch": "Lot", "LSL": 8.5, "HSL": 11.5,
              "goal": 10.0}
    model = Capability.Capability(df, mydict)
    fig = model.Report()
    blocks = _extract_cap_values(fig)
    (pp, ppl, ppu, ppk), (cp, cpl, cpu, cpk) = blocks[0], blocks[1]
    scalars = {
        "Pp": pp, "PpL": ppl, "PpU": ppu, "Ppk": ppk,
        "Cp": cp, "CpL": cpl, "CpU": cpu, "Cpk": cpk,
    }
    return {}, scalars, model.getLog()


def _tables():
    """Snapshot every Stat_Tables getter for n=2..25 and n>25 behavior."""
    # get_d2(n>25) uses a Monte-Carlo approximation based on np.random;
    # seed it so the snapshot is deterministic.
    np.random.seed(20240303)
    tbl = tables.Stat_Tables()
    getters = [
        "get_A", "get_A2", "get_A3", "get_c4", "get_B3", "get_B4", "get_B5",
        "get_B6", "get_d2", "get_d3", "get_D1", "get_D2", "get_D3", "get_D4",
    ]
    snap = {}
    for name in getters:
        fn = getattr(tbl, name)
        for n in range(2, 26):
            snap[f"{name}({n})"] = fn(n)
        # n > 25 without bol_Pass must raise; with bol_Pass (when the
        # signature supports it) returns the n=25 value. get_A, get_c4,
        # get_A3, get_B3..get_B6 compute an approximation instead.
        try:
            snap[f"{name}(26)"] = fn(26)
        except ValueError:
            snap[f"{name}(26)"] = "ValueError"
        try:
            snap[f"{name}(26,True)"] = fn(26, bol_Pass=True)
        except TypeError:
            snap[f"{name}(26,True)"] = "TypeError"
        except ValueError:
            snap[f"{name}(26,True)"] = "ValueError"
    return {}, snap, None


def _rnr_attribute():
    """Snapshot RnR.RnRAttribute aggregate dictionaries."""
    df = pd.read_csv(DATA / "Cube_surface.csv", sep=";")
    dict_key = {
        "1": "Operator",
        "2": "Pieze",
        "3": "Reference",
        "4": "Measurement",
    }
    model = RnR.RnRAttribute(mydf_Raw=df, mydict_key=dict_key)
    scalars = {
        "dict_Op": model._RnRAttribute__dict_Op,
        "dict_Sys": model._RnRAttribute__dict_Sys,
    }
    return {}, scalars, model.getLog()


# ---------------------------------------------------------------------------
# Serialization / comparison
# ---------------------------------------------------------------------------
def _df_to_text(df):
    buffer = io.StringIO()
    df.to_csv(buffer, float_format="%.17g")
    return buffer.getvalue()


def _text_to_df(text):
    return pd.read_csv(io.StringIO(text), index_col=0)


def _close(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return math.isclose(float(a), float(b), rel_tol=TOL, abs_tol=TOL)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _compare_frames(name, ref_text, new_text, errors):
    ref = _text_to_df(ref_text)
    new = _text_to_df(new_text)
    if list(ref.columns) != list(new.columns):
        errors.append(f"{name}: columns differ {ref.columns} vs {new.columns}")
        return
    if len(ref) != len(new):
        errors.append(f"{name}: row count differs {len(ref)} vs {len(new)}")
        return
    if list(ref.index.astype(str)) != list(new.index.astype(str)):
        errors.append(f"{name}: index differs")
        return
    for col in ref.columns:
        for i, (a, b) in enumerate(zip(ref[col], new[col])):
            if not _close(a, b):
                errors.append(
                    f"{name}: cell [{i}, {col!r}] differs: {a!r} vs {b!r}"
                )


def _compare_scalars(name, ref, new, errors):
    if isinstance(ref, dict):
        if set(ref) != set(new):
            errors.append(f"{name}: keys differ {set(ref)} vs {set(new)}")
            return
        for key in ref:
            _compare_scalars(f"{name}.{key}", ref[key], new[key], errors)
    elif not _close(ref, new):
        errors.append(f"{name}: {ref!r} vs {new!r}")


def _collect():
    """Build the whole snapshot. Returns (frames, scalars, logs)."""
    frames, scalars, logs = {}, {}, {}
    for prefix, builder in [
        ("rnr", _rnr_numeric),
        ("rep", _repeatability),
        ("cap_st", _capability_st),
        ("cap_lt", _capability_lt),
        ("tbl", _tables),
        ("attr", _rnr_attribute),
    ]:
        f, s, log = builder()
        frames.update(f)
        scalars[prefix] = s
        if log is not None:
            logs[f"{prefix}_log.txt"] = log
    return frames, scalars, logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["save", "compare"], help="save or compare"
    )
    args = parser.parse_args()

    frames, scalars, logs = _collect()

    if args.mode == "save":
        BASELINE.mkdir(exist_ok=True)
        for name, df in frames.items():
            (BASELINE / name).write_text(_df_to_text(df))
        (BASELINE / "scalars.json").write_text(
            json.dumps(scalars, indent=2, default=str)
        )
        for name, log in logs.items():
            (BASELINE / name).write_text(log)
        print(f"Baseline saved to {BASELINE}")
        return 0

    errors = []
    for name, df in frames.items():
        ref_path = BASELINE / name
        if not ref_path.exists():
            errors.append(f"{name}: missing reference file")
            continue
        _compare_frames(name, ref_path.read_text(), _df_to_text(df), errors)

    ref_scalars = json.loads((BASELINE / "scalars.json").read_text())
    _compare_scalars(
        "scalars", ref_scalars,
        json.loads(json.dumps(scalars, default=str)), errors,
    )

    for name, log in logs.items():
        ref_path = BASELINE / name
        if not ref_path.exists():
            errors.append(f"{name}: missing reference file")
            continue
        if ref_path.read_text() != log:
            errors.append(f"{name}: log text differs")

    if errors:
        print("BASELINE MISMATCH:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Baseline OK: all outputs match within 1e-9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
