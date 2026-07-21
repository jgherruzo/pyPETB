#           .---.        .-----------
#          /     \  __  /    ------
#         / /     \(..)/    -----
#        //////   ' \/ `   ---
#       //// / // :    : ---
#      // /   /  /`    '--
#     // /        //..\\
#   o===|========UU====UU=====-  -==========================o
#                '//||\\`
#                       DEVELOPED BY JGH
#
#   -=====================|===o  o===|======================-+

import matplotlib as mpl
import numpy as np
import pandas as pd
from pypetb import RnR, tables


# happy flow
def test_log():
    """Check log return a string."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)

    assert isinstance(RnRModel.getLog(), str) is True


def test_atlog():
    """Check log return a string."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/Cube_surface.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {
        "1": "Operator",
        "2": "Pieze",
        "3": "Reference",
        "4": "Measurement",
    }
    RnRModel = RnR.RnRAttribute(mydf_Raw=df, mydict_key=dict_key)

    assert isinstance(RnRModel.getLog(), str) is True


def test_RnRSolve_1():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.Total_Data, 2) == dict_Template["Total_Data"]


def test_RnRSolve_2():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.dbl_Range_avg, 2) == dict_Template["dbl_Range_avg"]


def test_RnRSolve_3():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.Total_avg, 2) == dict_Template["Total_avg"]


def test_RnRSolve_4():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.dbl_Range_UCL, 2) == dict_Template["dbl_Range_UCL"]


def test_RnRSolve_5():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.dbl_Avg_UCL, 2) == dict_Template["dbl_Avg_UCL"]


def test_RnRSolve_6():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.SStechnician, 2) == dict_Template["SStechnicia"]


def test_RnRSolve_7():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.SSpart, 2) == dict_Template["SSpart"]


def test_RnRSolve_8():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.SStotal, 2) == dict_Template["SStotal"]


def test_RnRSolve_9():
    """Check randomly RnRSolve Caltulation are made correctly."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    dict_Template = {
        "t": 3,
        "p": 10,
        "r": 3,
        "Total_Data": 90,
        "Total_min": -2.16,
        "Total_max": 2.26,
        "dbl_Range_avg": 0.34,
        "dbl_Range_UCL": 0.88,
        "dbl_Range_LCL": 0.0,
        "Total_avg": 0.0,
        "dbl_Avg_UCL": 0.35,
        "dbl_Avg_LCL": -0.35,
        "Sstd2": 0.11,
        "SStechnicia": 3.17,
        "Sstd2_": 9.82,
        "SSpart": 88.36,
        "SStotal": 94.65,
        "SSequipment": 2.76,
    }

    assert round(RnRModel.SSequipment, 2) == dict_Template["SSequipment"]


def test_RnRAnova_1():
    """Check RnRAnova return a pandas dataframe."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()
    assert isinstance(RnRModel.RnRAnova(), pd.core.frame.DataFrame) is True


def test_RnRAnova_2():
    """Check RnRAnova individual result."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()
    df = RnRModel.RnRAnova()
    assert round(df["MS"].loc["Part"], 2) == 9.82


def test_RnRAnova_3():
    """Check RnRAnova return  individual result."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()
    df = RnRModel.RnRAnova()
    assert round(df["F"].loc["Technician"], 2) == 79.41


def test_RnR_varTable_1():
    """Check variance table return a pandas dataframe."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    assert isinstance(RnRModel.RnR_varTable(), pd.core.frame.DataFrame) is True


def test_RnR_varTable_2():
    """Check variance table individual result."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()
    df = RnRModel.RnR_varTable()
    assert round(df["% Contribution"].loc["Technician"], 2) == 4.37


def test_RnR_varTable_3():
    """Check variance table individual result."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()
    df = RnRModel.RnR_varTable()
    assert round(df["Variance"].loc["Eq.Var. (Repeatability)"], 2) == 0.04


def test_RnR_SDTable():
    """Check SD table return a pandas dataframe."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    assert isinstance(RnRModel.RnR_SDTable(), pd.core.frame.DataFrame) is True


def test_RnR_RunChart():
    """Check if run chart returns a plt figure."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    RnRModel.RnRSolve()
    figure = RnRModel.RnR_RunChart()
    assert isinstance(figure, mpl.figure.Figure) is True


def test_RnR_Report():
    """Check if RnR_Report returns a plt figure."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    RnRModel.RnRSolve()

    assert isinstance(RnRModel.RnR_Report(), mpl.figure.Figure) is True


def test_RnR_Report2():
    """Check if supplied title to RnR_Report is applied."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    RnRModel.RnRSolve()
    report_title = "Test title"

    assert (
        RnRModel.RnR_Report(report_title).texts[0].get_text() == report_title
    )


def test_anova_computed_once_per_report(monkeypatch):
    """A5: the ANOVA pipeline is computed only once per report.

    A single RnRAnova computation performs exactly 3 f.cdf calls, so
    solving the model and requesting every table plus the full report
    must not increase that counter.
    """
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)

    calls = {"n": 0}
    real_cdf = RnR.f.cdf

    def counting_cdf(*args, **kwargs):
        calls["n"] += 1
        return real_cdf(*args, **kwargs)

    monkeypatch.setattr(RnR.f, "cdf", counting_cdf)
    RnRModel.RnRSolve()
    assert calls["n"] == 3
    RnRModel.RnRAnova()
    RnRModel.RnR_varTable()
    RnRModel.RnR_SDTable()
    RnRModel.RnR_Report()
    assert calls["n"] == 3


def test_anova_cache_returns_copies():
    """A5: cached tables are equal but independent objects."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    RnRModel.RnRSolve()

    df_a = RnRModel.RnRAnova()
    df_b = RnRModel.RnRAnova()
    assert df_a is not df_b
    assert df_a.equals(df_b)

    df_v1 = RnRModel.RnR_varTable()
    df_v2 = RnRModel.RnR_varTable()
    assert df_v1 is not df_v2
    assert df_v1.equals(df_v2)

    df_s1 = RnRModel.RnR_SDTable()
    df_s2 = RnRModel.RnR_SDTable()
    assert df_s1 is not df_s2
    assert df_s1.equals(df_s2)


def test_RnRSolve_range_control_limits_use_trial_count():
    """B5: Xbar-R chart subgroup size is the trial count (r), not the
    operator count (t).

    Each row of the range chart is the range of the r trial values
    measured by one operator on one part, so D4/D3/A2 must be looked up
    by r -- confirmed against the AIAG/SPC-for-Excel Xbar-R convention.
    Uses t=2 operators, r=3 trials so the two subgroup sizes disagree.
    """
    bases = [10.0, 12.0, 9.0]
    offsets = [0.00, 0.03, -0.02]
    rows = [
        {"Operator": op, "Part": str(part), "Measurement": base + offset}
        for op in ["A", "B"]
        for part, base in enumerate(bases)
        for offset in offsets
    ]
    df = pd.DataFrame(rows)
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    assert RnRModel.t == 2
    assert RnRModel.r == 3

    tbl = tables.Stat_Tables()
    expected_ucl = RnRModel.dbl_Range_avg * tbl.get_D4(3)
    assert abs(RnRModel.dbl_Range_UCL - expected_ucl) < 1e-9
    # guard against regressing to the pre-B5 operator-count indexing
    wrong_ucl = RnRModel.dbl_Range_avg * tbl.get_D4(2)
    assert abs(RnRModel.dbl_Range_UCL - wrong_ucl) > 1e-6


def test_RnRSolve_pivot_is_fast():
    """A4: vectorized run-pivot should solve 7200 rows well under 0.5s.

    Approximate, non-strict timing check: the previous row-by-row
    pd.concat implementation needed ~1s for this dataset (O(n^2)).
    """
    import time

    rng = np.random.default_rng(7)
    t, p, r = 4, 600, 3
    n = t * p * r
    df = pd.DataFrame(
        {
            "Op": np.repeat([f"O{i}" for i in range(t)], p * r),
            "Part": np.tile(np.repeat([f"P{i}" for i in range(p)], r), t),
            "Val": rng.normal(10, 1, n),
        }
    )
    dict_key = {"1": "Op", "2": "Part", "3": "Val"}
    start = time.perf_counter()
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5


def test_RnR_Report_many_operators():
    """B1: report builds with more operators than palette entries.

    Previously the random color-generation loop never terminated with
    8+ operators. Uses the Agg backend (headless test environment).
    """
    rng = np.random.default_rng(11)
    n_op, n_part, n_run = 9, 5, 2
    n = n_op * n_part * n_run
    df = pd.DataFrame(
        {
            "Op": np.repeat([f"O{i}" for i in range(n_op)], n_part * n_run),
            "Part": np.tile(
                np.repeat([f"P{i}" for i in range(n_part)], n_run), n_op
            ),
            "Val": rng.normal(10, 1, n),
        }
    )
    dict_key = {"1": "Op", "2": "Part", "3": "Val"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    RnRModel.RnRSolve()

    assert isinstance(RnRModel.RnR_Report(), mpl.figure.Figure) is True


def _figure_styles(figure):
    """Collect (color, marker) of every plotted line in a figure."""
    styles = []
    for ax in figure.axes:
        for line in ax.lines:
            styles.append((line.get_color(), line.get_marker()))
    return styles


def test_reports_are_reproducible():
    """A7: two consecutive reports use identical colors and markers."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    RnRModel.RnRSolve()

    assert _figure_styles(RnRModel.RnR_Report()) == _figure_styles(
        RnRModel.RnR_Report()
    )
    assert _figure_styles(RnRModel.RnR_RunChart()) == _figure_styles(
        RnRModel.RnR_RunChart()
    )


def test_attribute_system_concordance():
    """B2: System concordance accuracy uses the current part's data.

    Hand-calculated dataset (1 trial per operator and part):
      P1: both OK,  ref OK  -> concordant, matches
      P2: both OK,  ref BAD -> concordant, does not match
      P3: OK/BAD,   ref OK  -> not concordant
      P4: both BAD, ref BAD -> concordant, matches
    System repeatability = 3/4 = 75%, accuracy = 2/4 = 50%.
    The stale-variable bug evaluated the last operator's last part for
    every concordant part, yielding 75% accuracy here.
    """
    df = pd.DataFrame(
        {
            "Op": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "Part": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "Ref": ["OK", "OK", "BAD", "BAD", "OK", "OK", "BAD", "BAD"],
            "Val": ["OK", "OK", "OK", "OK", "OK", "BAD", "BAD", "BAD"],
        }
    )
    dict_key = {"1": "Op", "2": "Part", "3": "Ref", "4": "Val"}
    RnRModel = RnR.RnRAttribute(mydf_Raw=df, mydict_key=dict_key)

    dict_Op = RnRModel._RnRAttribute__dict_Op
    assert dict_Op["System"]["Rep"] == 75.0
    assert dict_Op["System"]["Acc"] == 50.0


# unhappy flow
def test_wrong_Column():
    """Check if wrong column is detected."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/RnR_Example.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Operator", "2": "Part", "3": "1"}

    try:
        RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    except ValueError as error:
        if "init_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_atwrong_Column():
    """Check if wrong column is detected."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/Cube_surface.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    mydict_key = {"1": "Operator", "2": "Pieze", "3": "Reference", "4": "1"}

    try:
        RnR.RnRAttribute(df, mydict_key)
    except ValueError as error:
        if "init_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_atless_Column():
    """Check if some column is not specified."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/Cube_surface.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    mydict_key = {"1": "Operator", "2": "Pieze", "3": "Reference"}

    try:
        RnR.RnRAttribute(df, mydict_key)
    except ValueError as error:
        if "init_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_nan_type():
    """Check if dataset contain any nan value"""
    lst_op = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"]
    lst_pieze = ["1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2", "2"]
    lst_value = [1, 1.1, 1.1, 1.05, 1, 1.1, np.nan, 1.1, 1.1, 1.05, 1, 1.1]
    df = pd.DataFrame()
    df["Operator"] = lst_op
    df["Part"] = lst_pieze
    df["Measurement"] = lst_value
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}

    try:
        RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    except ValueError as error:
        if "init_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_wrong_measurement():
    """Check if rnrsolve detect a different number of measurement."""
    lst_op = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    lst_pieze = ["1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2"]
    lst_value = [1, 1.1, 1.1, 1.05, 1, 1.1, 1, 1.1, 1.1, 1.05, 1]
    df = pd.DataFrame()
    df["Operator"] = lst_op
    df["Part"] = lst_pieze
    df["Measurement"] = lst_value
    dict_key = {"1": "Operator", "2": "Part", "3": "Measurement"}
    RnRModel = RnR.RnRNumeric(mydf_Raw=df, mydict_key=dict_key)
    try:
        RnRModel.RnRSolve()
    except ValueError as error:
        if "Solve_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True
