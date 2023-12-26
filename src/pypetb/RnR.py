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

import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypetb import tables as tbl
from scipy.stats import f


class RnRNumeric:
    """Repeatability and Reproducibility numeric gage analysis.
    RnRNumerics works as a model. It is defined using ameasurement dataframe.
    Log method could be called in order to check each parameter calculation.
    There is possibilities to get an anova table, standard deviation table or
    variance table, which are returned as pandas dataframe, or Report, where a
    matplotlib figure with the full analysis will be returned.
    Be sure to import matplotlib.pyplot in your script and write plt.show()
    after call .RnR_Report() or .Run_chart() to get the figures. Importind
    seaborn as sns and sns.set() is highly recommended to improve the sigth of
    the reports.

    Args:
    -------
    mydf_Raw : Pandas DataFrame containing 3 columns
        Operator, part and value

    mydict_key : Dictionary containing column names.
        key 1 to Operator column
        , key 2 for Part column
        , key 3 for Value column

    mydbl_tol : Optional. Int or Float
        system tolerance

    mydict_Info : Optional. Dictionary with info used on chart definition
        Report head info.
        . key 1 Measurement system name, key 2 Report Date
        , key 3 Engineer in charge of the study
        , key 4 Comments if needed or ''

    Methods:
    ----------
    getLog: string
          printable string containing all individual calculations

    RnRSolve: none
          determine all variables to make the RnR analysis

    RnRAnova: Pandas DataFrame
          two way anova

    RnR_varTable: Pandas DataFrame
          RnR variance analysis

    RnR_SDTable: Pandas DataFrame
          RnR standard deviation analysis

    RnR_RunChart: Figure
          Running chart for your measurement

    RnR_Report: Figure
          RnR whole report

    t: integer
          number of operator

    p: integer
          number of Piezes

    r: integer
          number of runs

    Total_Data: integer
          number of data in the input dataframe

    Total_min: float
          minimum measurement

    Total_max: float
          maximum measurement

    dbl_Range_avg: float
          average measures range

    dbl_Range_UCL: float
          Range upper control limit

    dbl_Range_LCL: float
          Range Lower control limit

    Total_avg: float
          whole measures average

    dbl_Avg_UCL: float
          average upper control limit

    dbl_Avg_LCL: float
          average lower control limit

    Sstd2: float
          Sum of deviation by operator

    SStechnician: float
          Total Operator Sum of deviation

    Sstd2_: float
          Sum of deviation by part

    SSpart: float
          Total Part Sum of deviation

    SStotal: float
          Total squared deviation

    SSequipment: float
          Equipment squared deviation

    Raises:
    ---------
    TypeError

    Init_01
        When input mydict_key contains a column which is not a mydf_Raw column
        name

    Init_02
        mydf_Raw contain nan values

    Init_03
        mydict_key['3'] is a non numerical type column

    Init_04
        mydbl_tol is not a number

    """

    def __init__(self, mydf_Raw, mydict_key, mydbl_tol=None, mydict_Info=None):
        """Initializate a new instance of a numeric RnR model"""
        self.__dict_key = mydict_key
        self.__dbl_tol = mydbl_tol
        if mydict_Info is None:
            mydict_Info = {"1": "", "2": "", "3": "", "4": ""}
        self.__dict_Info = mydict_Info

        if mydict_key["1"] not in mydf_Raw.keys().tolist():
            bol_key = True
            str_Column = mydict_key["1"]
            int_key = 1
        elif mydict_key["2"] not in mydf_Raw.keys().tolist():
            bol_key = True
            str_Column = mydict_key["2"]
            int_key = 2
        elif mydict_key["3"] not in mydf_Raw.keys().tolist():
            bol_key = True
            str_Column = mydict_key["3"]
            int_key = 3
        else:
            bol_key = False

        if self.__dbl_tol is not None:
            if (
                type(self.__dbl_tol) is not int
                and type(self.__dbl_tol) is not float
            ):
                raise ValueError(
                    f"Error init_04: Specified tolerance: {self.__dbl_tol} is"
                    " not a number"
                )
        if bol_key is True:
            raise ValueError(
                f"Error init_01: Specified dict_key: {int_key} |"
                f" {str_Column} is not in the original dataframe column names:"
                f" {mydf_Raw.keys().tolist()}, please, review your dict_key"
                " argument"
            )

        # Comprobamos tambien si hay algún dato nulo
        if mydf_Raw.isna().sum().sum() > 0:
            raise ValueError(
                "Error init_02: Given dataframe contains"
                f" {mydf_Raw.isna().sum().sum()} null values. Please, clean"
                " and filter your dataset before to call this method"
            )

        if pd.api.types.is_numeric_dtype(mydf_Raw[mydict_key["3"]]) is False:
            raise ValueError(
                f"Error init_03: Given dataframe column {mydict_key['3']} must"
                " be numeric type. Please, review your dataset"
            )

        # Create main working dataframe
        df_0 = pd.DataFrame()
        df_0["Op"] = mydf_Raw[mydict_key["1"]]
        df_0["Part"] = mydf_Raw[mydict_key["2"]]
        df_0["Valor"] = mydf_Raw[mydict_key["3"]]
        df_0["Op"] = df_0["Op"].astype("category")
        df_0["Part"] = df_0["Part"].astype("category")
        df_0["Valor"] = pd.to_numeric(df_0["Valor"])

        self.__log = list()
        self.__df_0 = df_0
        self.__log.append("Model is created")
        self.__Status = 1

    def getLog(self):
        """Return a string which contain all important calculations.

        Returns
        -------
        log: String
            all step logged
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")

        str_log = ""
        for i in range(0, len(self.__log)):
            str_log = str_log + self.__log[i] + "\n"

        return str_log

    def RnRSolve(self, bol_bias=False):
        """Calculate each individual value needed to make the RnR analysis and
        conclusions.

        Args:
        -------
        bol_bias : Optional. Boolean
            if bol_bias==False, RnRSolve will check if all
            piezes has the same number of runs and raise an error if not.
            If bol_bias==True then solve even if all piezes has no the same
            Number of runs

        Raises:
        ---------
        TypeError

        Solve_01
            if bol_bias==False get this error if some pieze has different
            number of runs
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")

        mydf_0 = self.__df_0

        if bol_bias is False:
            if (
                len(mydf_0.groupby("Part").count()["Valor"].unique().tolist())
                > 1
            ):
                raise ValueError(
                    """Error Solve_01: Some pieze has different
                                  number of runs. If would like to proceed,
                                  please, call .RnRSolve(bol_bias=True)"""
                )

        # One column per run
        df_temp1 = mydf_0[mydf_0["Op"] == mydf_0["Op"].unique()[0]]
        df_temp2 = df_temp1[df_temp1["Part"] == df_temp1["Part"].unique()[0]]

        # number of operator
        self.t = len(mydf_0["Op"].unique())
        # number of Piezes
        self.p = len(mydf_0["Part"].unique())
        # number of runs
        self.r = len(df_temp2)
        lst_columns = ["Run " + str(i) for i in range(0, self.r)]

        self.__log.append(
            "== DATASET EVALUATION ==\nOperator: {}\nTrials: {}\nPiezes: {}".format(  # noqa
                self.t, self.r, self.p
            )
        )

        # a dataframe containing a run per column is generated
        df = pd.DataFrame(columns=["OP", "Part"] + lst_columns)
        for op in mydf_0["Op"].unique():
            df_temp1 = mydf_0[mydf_0["Op"] == op]
            for part in df_temp1["Part"].unique():
                df_temp2 = df_temp1[df_temp1["Part"] == part]
                dict_value = dict()
                dict_value["OP"] = op
                dict_value["Part"] = part
                i = 0
                for value in range(0, len(df_temp2["Valor"])):
                    test = "Run " + str(i)
                    dict_value[test] = df_temp2["Valor"].iloc[value]
                    i += 1

                df_dictionary = pd.DataFrame([dict_value])
                df = pd.concat([df, df_dictionary], ignore_index=True)

        df.set_index(["OP", "Part"], inplace=True)

        df_1 = df.copy()
        df_1["Range"] = df.max(axis=1) - df.min(axis=1)
        df_1["Mean"] = df.mean(axis=1)

        # Create a dataframe goup by operator and another by part for
        # future calculations
        df_2 = df_1.groupby("OP").mean()
        df_3 = df_1.groupby("Part").mean()

        # Start making calculations
        self.__log.append("== CALCULATION ==")
        self.Total_Data = self.t * self.r * self.p
        self.__log.append("Total data: %.0f" % self.Total_Data)

        self.Total_max = np.max(df.max())
        self.__log.append("Max. measured value: %.4f" % self.Total_max)

        self.Total_min = np.min(df.min())
        self.__log.append("Min. measured value: %.4f" % self.Total_min)

        # Determine averange range, Range UCL and LCL
        self.dbl_Range_avg = df_1["Range"].mean()
        self.dbl_Range_UCL = self.dbl_Range_avg * tbl.get_D4(self.t)
        self.dbl_Range_LCL = self.dbl_Range_avg * tbl.get_D3(self.t)

        # Determine whole average, UCL and LCL
        self.Total_avg = np.mean(df.mean())
        self.__log.append("Avg. measured value: %.4f" % self.Total_avg)
        self.dbl_Avg_UCL = (
            self.Total_avg + tbl.get_A2(self.t) * self.dbl_Range_avg
        )
        self.dbl_Avg_LCL = (
            self.Total_avg - tbl.get_A2(self.t) * self.dbl_Range_avg
        )

        if self.dbl_Range_LCL < 0:
            self.dbl_Range_LCL = 0

        self.__log.append(
            "Avg. Control limits\nUCL: {:.4f}\nLCL: {:.4f}".format(
                self.dbl_Avg_UCL, self.dbl_Avg_LCL
            )
        )

        self.__log.append("Avg. Range measured: %.4f" % self.dbl_Range_avg)

        self.__log.append(
            "Range Control limits\nUCL: {:.4f}\nLCL: {:.4f}".format(
                self.dbl_Range_UCL, self.dbl_Range_LCL
            )
        )

        # 12/11/2023 10:03
        df_2["std2"] = (df_2["Mean"] - self.Total_avg) ** 2

        self.Sstd2 = df_2["std2"].sum()
        self.__log.append(
            "Sum of deviation by operator: {:.6f}".format(self.Sstd2)
        )

        self.SStechnician = self.p * self.r * self.Sstd2
        self.__log.append(
            "Total Operator Sum of deviation: {:.6f}".format(self.SStechnician)
        )

        df_3["std2"] = (df_3["Mean"] - self.Total_avg) ** 2

        self.Sstd2_ = df_3["std2"].sum()
        self.__log.append(
            "Sum of deviation by part: {:.6f}".format(self.Sstd2_)
        )

        self.SSpart = self.t * self.r * self.Sstd2_
        self.__log.append(
            "Total Part Sum of deviation: {:.6f}".format(self.SSpart)
        )

        lst_titles = list()
        lst_titlesw = list()
        lst_Run = list()
        for i in range(0, self.r):
            str_title = "sdev_Run " + str(i)
            str_titlew = "sdev_Run_w " + str(i)
            str_run = "Run " + str(i)
            df_1[str_title] = (df_1[str_run] - self.Total_avg) ** 2
            df_1[str_titlew] = (df_1[str_run] - df_1["Mean"]) ** 2
            lst_titles.append(str_title)
            lst_titlesw.append(str_titlew)
            lst_Run.append(str_run)

        Total_std2 = np.sum(df_1[lst_titles].sum())
        self.SStotal = Total_std2
        self.__log.append(
            "Total squared deviation: {:.6f}".format(self.SStotal)
        )

        self.SSequipment = np.sum(df_1[lst_titlesw].sum())
        self.__log.append(
            "Equipment squared deviation: {:.6f}".format(self.SSequipment)
        )

        self.SStechnicianxpart = Total_std2 - (
            self.SStechnician + self.SSpart + self.SSequipment
        )
        self.__log.append(
            "Iteration sum of squared: {:.6f}".format(self.SStechnicianxpart)
        )

        self.__df = df
        self.__df_3 = df_3
        self.__df_2 = df_2
        self.__df_1 = df_1

        self.__Status = 2

    def RnRAnova(self):
        """After calling .RnRSolve() anova analysis could be done.
        It will be returned as pandas DataFrame and all of the values will be
        accesibles from the dataframe.

        Returns
        -------
        Pandas DataFrame
            Anova result tabulated into a pandas dataframe
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")
        elif self.__Status == 1:
            raise ValueError(
                "You must call RnR.RnRSolve() before to call this method"
            )

        df_Anova = pd.DataFrame()
        df_Anova["Source of variability"] = [
            "Technician",
            "Part",
            "TechxPart (iteration)",
            "Repeatability with",
            "Repeatability without",
            "Total",
        ]
        df_Anova["DF"] = [
            (self.t - 1),
            (self.p - 1),
            ((self.t - 1) * (self.p - 1)),
            ((self.t * self.p) * (self.r - 1)),
            ((self.t * self.p) * (self.r - 1) + (self.t - 1) * (self.p - 1)),
            (self.t * self.p * self.r - 1),
        ]

        df_Anova["SS"] = [
            (self.SStechnician),
            (self.SSpart),
            (self.SStechnicianxpart),
            (self.SSequipment),
            (self.SSequipment + self.SStechnicianxpart),
            (self.SStotal),
        ]

        MStechnician = self.SStechnician / (self.t - 1)
        MSpart = self.SSpart / (self.p - 1)
        MStechnicianxPart = self.SStechnicianxpart / (
            (self.t - 1) * (self.p - 1)
        )
        MSequipment = self.SSequipment / (self.t * self.p * (self.r - 1))

        df_Anova["MS"] = [
            (MStechnician),
            (MSpart),
            (MStechnicianxPart),
            (MSequipment),
            (self.SSequipment + self.SStechnicianxpart)
            / ((self.t * self.p) * (self.r - 1) + (self.t - 1) * (self.p - 1)),
            (np.nan),
        ]

        df_Anova["F"] = [
            (MStechnician / MStechnicianxPart),
            (MSpart / MStechnicianxPart),
            ((MStechnicianxPart / MSequipment)),
            (np.nan),
            (np.nan),
            (np.nan),
        ]

        fdist = 1 - f.cdf(
            MStechnicianxPart / MSequipment,
            (self.t - 1) * (self.p - 1),
            self.t * self.p * (self.r - 1),
        )
        if fdist > 0.05:
            df_Anova["P"] = [
                (
                    (
                        1
                        - f.cdf(
                            MStechnician / MStechnicianxPart,
                            (self.t - 1),
                            (self.t - 1) * (self.p - 1),
                        )
                    )
                ),
                (
                    1
                    - f.cdf(
                        MSpart / MStechnicianxPart,
                        (self.p - 1),
                        (self.t - 1) * (self.p - 1),
                    )
                ),
                fdist,
                (np.nan),
                fdist,
                (np.nan),
            ]
        else:
            df_Anova["P"] = [
                (
                    (
                        1
                        - f.cdf(
                            MStechnician / MStechnicianxPart,
                            (self.t - 1),
                            (self.t - 1) * (self.p - 1),
                        )
                    )
                ),
                (
                    1
                    - f.cdf(
                        MSpart / MStechnicianxPart,
                        (self.p - 1),
                        (self.t - 1) * (self.p - 1),
                    )
                ),
                fdist,
                fdist,
                (np.nan),
                (np.nan),
            ]

        df_Anova.set_index("Source of variability", inplace=True)
        return df_Anova

    def RnR_varTable(self):
        """After calling .RnRSolve() variance table could be done.
        It will be returned as pandas DataFrame and all of the values will be
        accesibles from the dataframe.

        Returns
        -------
        Pandas DataFrame
            Variante table result tabulated into a pandas dataframe
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")
        elif self.__Status == 1:
            raise ValueError(
                "You must call RnR.RnRSolve() before to call this method"
            )

        df_Anova = self.RnRAnova()
        dbl_fdist = df_Anova.loc["TechxPart (iteration)"]["P"]
        if dbl_fdist <= 0.05:
            Srpeatability = self.SSequipment / (self.t * self.p * (self.r - 1))
            MStechnicianxPart = df_Anova.loc["TechxPart (iteration)"]["MS"]
        else:
            Srpeatability = (self.SSequipment + self.SStechnicianxpart) / (
                (self.t * self.p) * (self.r - 1) + (self.t - 1) * (self.p - 1)
            )
            MStechnicianxPart = Srpeatability

        StechniciaxPart = (MStechnicianxPart - Srpeatability) / self.r
        if StechniciaxPart < 0:
            StechniciaxPart = 0

        MSpart = df_Anova.loc["Part"]["MS"]
        Spart = (MSpart - MStechnicianxPart) / (self.r * self.t)
        MStechnician = df_Anova.loc["Technician"]["MS"]
        Stechnician = (MStechnician - MStechnicianxPart) / (self.r * self.p)
        if Stechnician < 0:
            Stechnician = 0

        GRnR = Srpeatability + Stechnician
        # EV = Srpeatability
        OV = Stechnician + StechniciaxPart
        PtP = Spart
        TV = OV + Srpeatability + PtP
        df_varTbl = pd.DataFrame()
        df_varTbl["Source"] = [
            "Total Gage R&R",
            "Eq.Var. (Repeatability)",
            "Op.Var. (Reproducibility)",
            "Technician",
            "Technician x Part iter.",
            "Part to Part",
            "Total variation",
        ]
        df_varTbl["Variance"] = [
            (GRnR),
            (Srpeatability),
            (OV),
            (Stechnician),
            (StechniciaxPart),
            (PtP),
            (TV),
        ]
        df_varTbl["% Contribution"] = [
            (GRnR / TV * 100),
            (Srpeatability / TV * 100),
            (OV / TV * 100),
            (Stechnician / TV * 100),
            (StechniciaxPart / TV * 100),
            (PtP / TV * 100),
            (TV / TV * 100),
        ]
        df_varTbl.set_index("Source", inplace=True)
        return df_varTbl

    def RnR_SDTable(self):
        """After calling .RnRSolve() standard deviation table could be done.

        It
        will be returned as pandas DataFrame and all of the values will be
        accesibles from the dataframe.

        Returns
        -------
        Pandas DataFrame
            Standard deviation table result tabulated into a pandas dataframe
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")
        elif self.__Status == 1:
            raise ValueError(
                "You must call RnR.RnRSolve() before to call this method"
            )
        df_varTbl = self.RnR_varTable()
        GRnR = df_varTbl.loc["Total Gage R&R"]["Variance"]
        EV = df_varTbl.loc["Eq.Var. (Repeatability)"]["Variance"]
        OV = df_varTbl.loc["Op.Var. (Reproducibility)"]["Variance"]
        PtP = df_varTbl.loc["Part to Part"]["Variance"]
        TV = df_varTbl.loc["Total variation"]["Variance"]

        Stechnician = df_varTbl.loc["Technician"]["Variance"]
        StechniciaxPart = df_varTbl.loc["Technician x Part iter."]["Variance"]

        df_SDTbl = pd.DataFrame()
        df_SDTbl["Source"] = [
            "Total Gage R&R",
            "Eq.Var. (Repeatability)",
            "Op.Var. (Reproducibility)",
            "Technician",
            "Technician x Part iter.",
            "Part to Part",
            "Total variation",
        ]
        df_SDTbl["StdDev (SD)"] = [
            (GRnR ** (1 / 2)),
            (EV ** (0.5)),
            (OV ** (0.5)),
            (Stechnician ** (0.5)),
            (StechniciaxPart ** (0.5)),
            (PtP ** (0.5)),
            (TV ** (0.5)),
        ]
        df_SDTbl["StudyVar (6*SD)"] = [
            (6 * GRnR ** (1 / 2)),
            (6 * EV ** (0.5)),
            (6 * OV ** (0.5)),
            (6 * Stechnician ** (0.5)),
            (6 * StechniciaxPart ** (0.5)),
            (6 * PtP ** (0.5)),
            (6 * TV ** (0.5)),
        ]
        df_SDTbl["% Study Var"] = [
            (GRnR ** (1 / 2) / TV ** (0.5) * 100),
            (EV ** (0.5) / TV ** (0.5) * 100),
            (OV ** (0.5) / TV ** (0.5) * 100),
            (Stechnician ** (0.5) / TV ** (0.5) * 100),
            (StechniciaxPart ** (0.5) / TV ** (0.5) * 100),
            (PtP ** (0.5) / TV ** (0.5) * 100),
            (TV ** (0.5) / TV ** (0.5) * 100),
        ]
        df_SDTbl.set_index("Source", inplace=True)

        if self.__dbl_tol is not None:
            df_SDTbl["% tol (VE/tol)"] = [
                (6 * GRnR ** (1 / 2)) / self.__dbl_tol * 100,
                (6 * EV ** (0.5)) / self.__dbl_tol * 100,
                (6 * OV ** (0.5)) / self.__dbl_tol * 100,
                (6 * Stechnician ** (0.5)) / self.__dbl_tol * 100,
                (6 * StechniciaxPart ** (0.5)) / self.__dbl_tol * 100,
                (6 * PtP ** (0.5)) / self.__dbl_tol * 100,
                (6 * TV ** (0.5)) / self.__dbl_tol * 100,
            ]

        return df_SDTbl

    def RnR_RunChart(self):
        """Run chart is a figure that contain a chart per pieze where all the
        measurement made by the operator are showed.

        Returns
        -------
        matplotlib figure
            Set of charts
        """
        Fig1 = plt.figure(figsize=(18, 12))
        Fig1.set_facecolor("white")
        gs = mpl.gridspec.GridSpec(
            int((self.p / 5) + 1), 5, wspace=0, hspace=0.1
        )
        Fig1.suptitle(
            "\nRUN CHART FOR MEASUREMENT SYSTEM OF {} BY {}, {}".format(
                self.__dict_key["3"],
                self.__dict_key["2"],
                self.__dict_key["1"],
            ),
            fontsize=20,
        )

        # First row will be focused on info about the analysis
        lst_TT = list()
        lst_TT.append(Fig1.add_subplot(gs[0, :2]))  # first row for text
        lst_TT.append(Fig1.add_subplot(gs[0, 2]))  # first row for text
        lst_TT.append(Fig1.add_subplot(gs[0, 3:]))  # first row for text
        for i in range(0, len(lst_TT)):
            lst_TT[i].set_xticks([])
            lst_TT[i].set_yticks([])
            lst_TT[i].set_facecolor("white")
            lst_TT[i].set_ylim(0, 1)
            lst_TT[i].set_xlim(0, 1)

        lst_TT[0].annotate(
            "Measurement system name: {}\n\nDate: {}".format(
                self.__dict_Info["1"], self.__dict_Info["2"]
            ),
            xy=(0.1, 0.7),
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=12,
        )

        lst_TT[0].annotate(
            "CHART CALIBRATION\nMax. {:.3f}\nAvg. {:.3f}\nMin. {:.3f}".format(
                self.Total_max, self.Total_avg, self.Total_min
            ),
            xy=(0.1, 0.2),
            bbox=dict(boxstyle="round", fc="w", color="lightgrey"),
            fontsize=12,
        )

        lst_TT[1].annotate(
            "Panel var: {}".format(self.__dict_key["2"]),
            xy=(0, 0.1),
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=12,
            fontstyle="italic",
        )
        lst_TT[2].annotate(
            "Notified by: {}\n\nTolerance: {}\n\nMis: {}".format(
                self.__dict_Info["3"], self.__dbl_tol, self.__dict_Info["4"]
            ),
            xy=(0, 0.7),
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=12,
        )

        # initializate values for char drawing
        row = 1
        column = 0
        lst_ax = list()

        # Chose operator parameters
        dict_OperatorLine = dict()
        for i in range(0, len(self.__df.index.unique(level="OP"))):
            dict_OperatorLine[self.__df.index.unique(level="OP")[i]] = {
                "Color": random.choice(list(mpl.colors.CSS4_COLORS.values())),
                "Marker": random.choice(list(mpl.lines.Line2D.markers.keys())),
            }
        # print(dict_OperatorLine)

        # Create run chart
        for counter in range(0, self.p):
            # One box per pieze
            str_piece = self.__df.index.unique(level="Part")[counter]
            df_temp = self.__df.xs(str_piece, level=1, drop_level=False)
            lst_ax.append(Fig1.add_subplot(gs[row, column]))
            lst_ax[counter].set_title(str_piece, fontweight="bold")
            lst_ax[counter].yaxis.set_major_locator(plt.MaxNLocator(3))
            lst_ax[counter].set_yticks([])
            lst_ax[counter].set_xticks([])
            lst_ax[counter].set_ylim(self.Total_min, self.Total_max)
            column += 1
            if column > 4:
                row += 1
                column = 0
            # per pieze three blocks
            t_base = 0
            for key in dict_OperatorLine:
                x = np.arange(t_base, t_base + len(range(0, self.t)))
                str_op = key
                y = df_temp.xs(
                    str_op, level=0, drop_level=False
                ).values.flatten()
                t_base = t_base + len(range(0, self.t))
                lst_ax[counter].plot(
                    x,
                    y,
                    label=str_op,
                    color=dict_OperatorLine[key]["Color"],
                    marker=dict_OperatorLine[key]["Marker"],
                )
                lst_ax[counter].axhline(
                    self.Total_avg, color="black", linestyle="--"
                )

        lst_ax[4].legend(
            loc="upper right",
            bbox_to_anchor=(0.6, 1.7),
            title="{}".format(self.__dict_key["1"]),
        )
        return Fig1

    def RnR_Report(self):
        """RnR_Report chart is a figure that contain six important chart in
        order to conclude the status of the measurement system First chart will
        show the impact of each parameter that affect to the variation of the
        measurement. Second one shows a point chart in order to detect the
        measurement of each pieze. In the third one, average measures range is
        showed, while the fourth one is a violin plot of how measure each
        operator. The fith one shows the average measure per pieze and operator
        and the last one shows the average value per pieze measured by each
        operator.
        Trend color are ramdon and sometimes could be low visible, just repeat
        the command to change it.

        Returns
        -------
        matplotlib figure
            Set of charts
        """
        # df = self.__df
        df_0 = self.__df_0
        df_3 = self.__df_3
        # df_2 = self.__df_2
        df_1 = self.__df_1

        Fig2 = plt.figure(figsize=(16, 12))
        Fig2.set_facecolor("white")

        int_Chart_column = (self.t + 1) * 2
        gs = mpl.gridspec.GridSpec(4, int_Chart_column, wspace=0, hspace=0.6)
        Fig2.suptitle(
            "R&R Measurement System Report (ANOVA) for {}".format(
                self.__dict_key["3"]
            ),
            fontsize=20,
        )

        # ============================================================================================
        #                                HEAD
        # ============================================================================================
        lst_TT = list()
        lst_TT.append(
            Fig2.add_subplot(gs[0, : int(int_Chart_column / 2)])
        )  # first row for text
        lst_TT.append(
            Fig2.add_subplot(gs[0, int((int_Chart_column / 2) + 1) :])
        )  # first row for text
        for i in range(0, len(lst_TT)):
            lst_TT[i].set_xticks([])
            lst_TT[i].set_yticks([])
            lst_TT[i].set_facecolor("white")
            lst_TT[i].set_ylim(0, 1)
            lst_TT[i].set_xlim(0, 1)

        lst_TT[0].annotate(
            "Measurement system name: {}\n\nDate: {}".format(
                self.__dict_Info["1"], self.__dict_Info["2"]
            ),
            xy=(0.1, 0.7),
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=12,
        )

        lst_TT[1].annotate(
            "Notified by: {}\n\nTolerance: {}\n\nMis: {}".format(
                self.__dict_Info["3"], self.__dbl_tol, self.__dict_Info["4"]
            ),
            xy=(0, 0.7),
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=12,
        )

        # ============================================================================================
        #                                VARIACION
        # ============================================================================================
        df_varTbl = self.RnR_varTable()
        GRnR = df_varTbl.loc["Total Gage R&R"]["Variance"]
        EV = df_varTbl.loc["Eq.Var. (Repeatability)"]["Variance"]
        OV = df_varTbl.loc["Op.Var. (Reproducibility)"]["Variance"]
        PtP = df_varTbl.loc["Part to Part"]["Variance"]
        TV = df_varTbl.loc["Total variation"]["Variance"]
        # Stechnician = df_varTbl.loc["Technician"]["Variance"]
        # StechniciaxPart = df_varTbl
        #           .loc["Technician x Part iter."]["Variance"]

        x = ["RnR", "Repea.", "Repro.", "PtP"]
        y = {
            "% Contribution": (
                GRnR / TV * 100,
                EV / TV * 100,
                OV / TV * 100,
                PtP / TV * 100,
            ),
            "% Var. Study": (
                GRnR ** (1 / 2) / TV ** (0.5) * 100,
                EV ** (0.5) / TV ** (0.5) * 100,
                OV ** (0.5) / TV ** (0.5) * 100,
                PtP ** (0.5) / TV ** (0.5) * 100,
            ),
        }

        if self.__dbl_tol is not None:
            y.update(
                {
                    "% tol (VE/tol)": (
                        (6 * GRnR ** (1 / 2)) / self.__dbl_tol * 100,
                        (6 * EV ** (0.5)) / self.__dbl_tol * 100,
                        (6 * OV ** (0.5)) / self.__dbl_tol * 100,
                        (6 * PtP ** (0.5)) / self.__dbl_tol * 100,
                    )
                }
            )

        # print(y)

        ax1 = Fig2.add_subplot(gs[1, : int((int_Chart_column / 2) - 1)])
        X = np.arange(len(x))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for attribute, measurement in y.items():
            offset = width * multiplier
            # rects = ax1.bar(X + offset, measurement, width, label=attribute)
            ax1.bar(X + offset, measurement, width, label=attribute)
            # ax1.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax1.set_ylabel("Percentage")
        ax1.set_title("Variation Component", fontweight="bold")
        ax1.set_xticks(X)
        ax1.set_xticklabels(x)
        ax1.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
        ax1.set_ylim(0, 100)

        # ============================================================================================
        #                                Value per Piece
        # ============================================================================================
        ax2 = Fig2.add_subplot(gs[1, int((int_Chart_column / 2) + 1) :])
        ax2.set_title(
            "{} by {}".format(self.__dict_key["3"], self.__dict_key["2"]),
            fontweight="bold",
        )
        ax2.scatter(df_0["Part"], df_0["Valor"])
        ax2.plot(df_3.index, df_3["Mean"], color="orange")
        ax2.set_xlabel("{}".format(self.__dict_key["2"]))
        ax2.set_ylabel("{}".format(self.__dict_key["3"]))
        ax2.set_xticks(df_3.index)
        # ============================================================================================
        #                                R per Operator
        # ============================================================================================
        lst_ax3 = list()
        ax_max = max(df_1["Range"].max(), self.dbl_Range_UCL) * 1.1
        ax_min = min(df_1["Range"].min(), self.dbl_Range_UCL) - ax_max * 0.1
        for i in range(0, len(df_0["Op"].unique())):
            lst_ax3.append(Fig2.add_subplot(gs[2, i]))
            if i > 0:
                lst_ax3[i].set_yticks([])
            else:
                lst_ax3[i].set_ylabel("Sample Range")
            lst_ax3[i].set_title("{}".format(df_0["Op"].unique()[i]))
            lst_ax3[i].set_ylim(ax_min, ax_max)
            lst_ax3[i].axhline(
                self.dbl_Range_UCL,
                color="red",
                label="UCL={:.4f}".format(self.dbl_Range_UCL),
            )
            lst_ax3[i].axhline(
                self.dbl_Range_avg,
                color="black",
                linestyle="--",
                label="avg={:.4f}".format(self.dbl_Range_avg),
            )
            lst_ax3[i].axhline(
                self.dbl_Range_LCL,
                color="red",
                label="LCL={:.4f}".format(self.dbl_Range_LCL),
            )

            df_temp = df_1.xs(
                df_0["Op"].unique()[i], level=0, drop_level=False
            )
            c = random.choice(list(mpl.colors.CSS4_COLORS.values()))
            m = random.choice(list(mpl.lines.Line2D.markers.keys()))
            lst_ax3[i].plot(
                df_temp.index.get_level_values(1),
                df_temp["Range"],
                color=c,
                marker=m,
            )
            lst_ax3[i].set_xticks(df_temp.index.get_level_values(1))

        # lst_ax3[0].title.set_text('Range by {}'.format(dict_key['Op'])) #
        lst_ax3[1].set_xlabel("{}".format(self.__dict_key["2"]))  #
        lst_ax3[-1].legend(loc="upper right", bbox_to_anchor=(2.1, 1))
        # ============================================================================================
        #                                Violin Plot
        # ============================================================================================
        ax4 = Fig2.add_subplot(gs[2, int((int_Chart_column / 2) + 1) :])
        ax4.set_title(
            "{} by {}".format(self.__dict_key["3"], self.__dict_key["1"]),
            fontweight="bold",
        )
        for n in range(0, len((df_0["Op"].unique()))):
            c = random.choice(list(mpl.colors.CSS4_COLORS.values()))
            ax4.boxplot(
                df_0[df_0["Op"] == df_0["Op"].unique()[n]]["Valor"],
                positions=[n + 1],
                patch_artist=True,
                boxprops=dict(facecolor=c, color=c),
            )
        ax4.set_ylabel("{}".format(self.__dict_key["3"]))
        ax4.set_xticklabels(df_0["Op"].unique())
        # ============================================================================================
        #                                Xbarra per Operator
        # ============================================================================================
        lst_ax5 = list()
        ax5_max = self.Total_max * 1.1
        ax5_min = self.Total_min - self.Total_max * 0.1

        for i in range(0, len(df_0["Op"].unique())):
            lst_ax5.append(Fig2.add_subplot(gs[3, i]))
            if i > 0:
                lst_ax5[i].set_yticks([])
            else:
                lst_ax5[i].set_ylabel("Sample Avg")
            lst_ax5[i].set_title("{}".format(df_0["Op"].unique()[i]))
            lst_ax5[i].set_ylim(ax5_min, ax5_max)

            lst_ax5[i].axhline(
                self.dbl_Avg_UCL,
                color="red",
                label="UCL={:.4f}".format(self.dbl_Avg_UCL),
            )
            lst_ax5[i].axhline(
                self.Total_avg,
                color="black",
                linestyle="--",
                label="avg={:.4f}".format(self.Total_avg),
            )
            lst_ax5[i].axhline(
                self.dbl_Avg_LCL,
                color="red",
                label="LCL={:.4f}".format(self.dbl_Avg_LCL),
            )

            df_temp = df_1.xs(
                df_0["Op"].unique()[i], level=0, drop_level=False
            )
            c = random.choice(list(mpl.colors.CSS4_COLORS.values()))
            m = random.choice(list(mpl.lines.Line2D.markers.keys()))
            lst_ax5[i].plot(
                df_temp.index.get_level_values(1),
                df_temp["Mean"],
                color=c,
                marker=m,
            )
            lst_ax5[i].set_xticks(df_temp.index.get_level_values(1))

        lst_ax5[1].set_xlabel("{}".format(self.__dict_key["2"]))  #
        lst_ax5[-1].legend(loc="upper right", bbox_to_anchor=(2.1, 1))
        # ============================================================================================
        #                                Iteration
        # ============================================================================================
        ax6 = Fig2.add_subplot(
            gs[
                3,
                int((int_Chart_column / 2) + 1) : int((int_Chart_column - 1)),
            ]
        )
        ax6.set_title(
            "Iteration {}x{}".format(
                self.__dict_key["2"], self.__dict_key["1"]
            ),
            fontweight="bold",
        )
        for item in df_0["Op"].unique():
            df_temp = df_1.xs(item, level=0, drop_level=False)
            ax6.plot(
                df_temp.index.get_level_values(1), df_temp["Mean"], label=item
            )
        ax6.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
        ax6.set_xlabel("{}".format(self.__dict_key["2"]))
        ax6.set_ylabel("Mean")

        return Fig2
