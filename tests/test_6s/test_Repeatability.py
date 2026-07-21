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
from pypetb import Repeatability


# happy flow
def test_log():
    """Check log return a string."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)

    assert isinstance(RModel.getLog(), str) is True


def test_RAnova_1():
    """Check RnRAnova return a pandas dataframe."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()
    assert isinstance(RModel.RAnova(), pd.core.frame.DataFrame) is True


def test_R_varTable_1():
    """Check variance table return a pandas dataframe."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()

    assert isinstance(RModel.R_varTable(), pd.core.frame.DataFrame) is True


def test_R_SDTable():
    """Check SD table return a pandas dataframe."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()

    assert isinstance(RModel.R_SDTable(), pd.core.frame.DataFrame) is True


def test_R_RunChart():
    """Check if run chart returns a plt figure."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()
    figure = RModel.R_RunChart()
    assert isinstance(figure, mpl.figure.Figure) is True


def test_R_Report():
    """Check if RnR_Report returns a plt figure."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()

    assert isinstance(RModel.R_Report(), mpl.figure.Figure) is True


def test_anova_cache_returns_copies():
    """A5: ANOVA is cached by RSolve and tables return copies."""
    url = "https://raw.githubusercontent.com/markwkiehl/public_datasets/main/GR%26R%206_28_24%20perp1.csv"  # noqa\n
    df = pd.read_csv(url, sep=";")
    dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()

    assert RModel._RNumeric__df_anova is not None

    df_a = RModel.RAnova()
    df_b = RModel.RAnova()
    assert df_a is not df_b
    assert df_a.equals(df_b)

    df_v1 = RModel.R_varTable()
    df_v2 = RModel.R_varTable()
    assert df_v1 is not df_v2
    assert df_v1.equals(df_v2)

    df_s1 = RModel.R_SDTable()
    df_s2 = RModel.R_SDTable()
    assert df_s1 is not df_s2
    assert df_s1.equals(df_s2)


def test_RSolve_pivot_is_fast():
    """A4: vectorized run-pivot should solve 7200 rows well under 0.5s.

    Approximate, non-strict timing check: the previous row-by-row
    pd.concat implementation needed ~1s for this dataset (O(n^2)).
    """
    import time

    rng = np.random.default_rng(7)
    r = 3
    n = 2400 * r
    df = pd.DataFrame(
        {
            "Part": np.repeat([f"P{i}" for i in range(2400)], r),
            "Val": rng.normal(10, 1, n),
        }
    )
    dict_key = {"1": "Part", "2": "Val"}
    start = time.perf_counter()
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    RModel.RSolve()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5


# unhappy flow
def test_nan_type():
    """Check if RnR_Report returns a plt figure."""
    lst_pieze = ["1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2", "2"]
    lst_value = [1, 1.1, 1.1, 1.05, 1, 1.1, np.nan, 1.1, 1.1, 1.05, 1, 1.1]
    df = pd.DataFrame()
    df["Part"] = lst_pieze
    df["Measurement"] = lst_value
    dict_key = {"1": "Part", "2": "Measurement"}

    try:
        Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key, mydbl_tol=8)
    except ValueError as error:
        if "init_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_wrong_measurement():
    """Check if rnrsolve detect a different number of measurement."""
    lst_pieze = ["1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2"]
    lst_value = [1, 1.1, 1.1, 1.05, 1, 1.1, 1, 1.1, 1.1, 1.05, 1]
    df = pd.DataFrame()
    df["Part"] = lst_pieze
    df["Measurement"] = lst_value
    dict_key = {"1": "Part", "2": "Measurement"}
    RModel = Repeatability.RNumeric(mydf_Raw=df, mydict_key=dict_key)
    try:
        RModel.RSolve()
    except ValueError as error:
        if "Solve_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True
