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
from pypetb import Capability


def CpString():
    """Test correct input data type"""
    arr_temp = np.chararray(20)
    arr_temp[:] = "a"
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": 3, "HSL": "", "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
    except ValueError as error:
        # print(str(error))
        if "init_04" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def CpNull():
    """Test correct input data type"""
    arr_temp = np.array([45, 7, 67, np.nan])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": 3, "HSL": "", "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
    except ValueError as error:
        # print(str(error))
        if "init_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def CpWrongDict():
    """Test correct input dictionary"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"values": 0, "batch": "", "LSL": 3, "HSL": "", "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
    except ValueError as error:
        # print(str(error))
        if "init_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def CpWrongColumn():
    """Test correct input dictionary"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": "0", "batch": "", "LSL": 3, "HSL": "", "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
    except ValueError as error:
        # print(str(error))
        if "init_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def CpNoTol():
    """Test correct input dictionary"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": "", "HSL": "", "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
    except ValueError as error:
        # print(str(error))
        if "init_05" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def CpLSL():
    """Test correct input dictionary"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": 1, "HSL": "", "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
        bol_temp = True
    except ValueError as error:  # noqa : F841
        bol_temp = False

    assert bol_temp is True


def CpHSL():
    """Test correct input dictionary"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": "1", "HSL": 1, "goal": "*"}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
        bol_temp = True
    except ValueError as error:  # noqa : F841
        bol_temp = False

    assert bol_temp is True


def Cpgoal():
    """Test correct input dictionary"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": "1", "HSL": 1, "goal": 1}
    try:
        Model_Cp = Capability.Capability(df, dict_info)  # noqa : F841
        bol_temp = True
    except ValueError as error:  # noqa : F841
        bol_temp = False

    assert bol_temp is True


def test_bool_limits():
    """B8: booleans are not valid numeric limits (LSL/HSL/goal)."""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": True, "HSL": "",
                 "goal": "*"}
    try:
        Capability.Capability(df, dict_info)
        raise AssertionError("ValueError was not raised")
    except ValueError as error:
        assert "init_05" in str(error)

    # a boolean HSL must be ignored too: with HSL=True and LSL=""
    # there is no numeric limit left
    dict_info = {"value": 0, "batch": "", "LSL": "", "HSL": True,
                 "goal": "*"}
    try:
        Capability.Capability(df, dict_info)
        raise AssertionError("ValueError was not raised")
    except ValueError as error:
        assert "init_05" in str(error)


def test_log():
    """Check log return a string."""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": "1", "HSL": 1, "goal": 1}
    Model_Cp = Capability.Capability(df, dict_info)

    assert isinstance(Model_Cp.getLog(), str) is True


def test_Normality():
    """Check if Normality_test returns a plt figure."""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": "1", "HSL": 1, "goal": 1}
    Model_Cp = Capability.Capability(df, dict_info)
    figure = Model_Cp.Normality_test()
    assert isinstance(figure, mpl.figure.Figure) is True


def test_Normality_probplot_axis():
    """B6: the Q-Q plot is drawn on the report's own second subplot."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(5)
    df = pd.DataFrame({"Meas": rng.normal(10, 1, 60)})
    dict_info = {"value": "Meas", "batch": "", "LSL": 7, "HSL": 13,
                 "goal": 10}
    Model_Cp = Capability.Capability(df, dict_info)

    fignums_before = len(plt.get_fignums())
    figure = Model_Cp.Normality_test()

    # only the report figure itself is registered with pyplot
    assert len(plt.get_fignums()) == fignums_before + 1
    # the Q-Q subplot (histogram, probplot, series, descriptive order)
    # contains the probability-plot fit line and sample markers
    assert len(figure.axes) == 4
    assert len(figure.axes[1].lines) >= 2


def Cp_Report():
    """Check if Report returns a plt figure for short term"""
    arr_temp = np.array([45, 7, 67, 1])
    df = pd.DataFrame(arr_temp)
    dict_info = {"value": 0, "batch": "", "LSL": "1", "HSL": 1, "goal": 1}
    Model_Cp = Capability.Capability(df, dict_info)
    figure = Model_Cp.Report()
    assert isinstance(figure, mpl.figure.Figure) is True


def Pp_Report():
    """Check if Report returns a plt figure for long term"""
    arr_temp = np.array([45, 7, 67, 1])
    arr_temp_id = np.array([21, 21, 1, 1])
    df = pd.DataFrame()
    df["0"] = arr_temp
    df["1"] = arr_temp_id
    dict_info = {"value": "0", "batch": "1", "LSL": "1", "HSL": 1, "goal": 1}

    Model_Cp = Capability.Capability(df, dict_info)
    figure = Model_Cp.Report()
    assert isinstance(figure, mpl.figure.Figure) is True
