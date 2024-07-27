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


# unhappy flow
def test_nan_type():
    """Check if RnR_Report returns a plt figure."""
    lst_pieze = ["1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2", "2"]
    lst_value = [1, 1.1, 1.1, 1.05, 1, 1.1, np.NaN, 1.1, 1.1, 1.05, 1, 1.1]
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
