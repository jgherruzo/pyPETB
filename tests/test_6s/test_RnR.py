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
from pypetb import RnR


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


def test_at_Report():
    """Check if RnR_Report returns a plt figure."""
    url = "https://raw.githubusercontent.com/jgherruzo/myFreeDatasets/main/Cube_surface.csv"  # noqa
    df = pd.read_csv(url, sep=";")
    mydict_key = {
        "1": "Operator",
        "2": "Pieze",
        "3": "Reference",
        "4": "Measurement",
    }
    RnRModel = RnR.RnRAttribute(df, mydict_key)
    RnRModel.Report()

    assert isinstance(RnRModel.Report(), mpl.figure.Figure) is True


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
    lst_value = [1, 1.1, 1.1, 1.05, 1, 1.1, np.NaN, 1.1, 1.1, 1.05, 1, 1.1]
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
