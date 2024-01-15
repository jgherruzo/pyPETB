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
import statistics

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from scipy import stats


class Capability:
    """Capability analysis
    Capability works as a model

    Args:
    -----
    mydf : pandas dataframe
      df must contain at least the measures in a column. For long term
        analysis, batch ID could be added

    mydict : dictionary
      Value: String. Measures column name | Batch: Optional String. Batch ID
        column name | LSL: Optional float. Lower specification limit | HSL:
          Optional float. Higher specification limit | Goal: Optional float.
            Goal. ONE OF LSL OR HSL MUST BE SPECIFIED

    Methods:
    ----------
    getLog: string
          printable string containing all individual calculations

    Normality_test: matplotlib figure
          report to check sample normality

    Report: matplotlib figure
          Capability report

    Raises:
    ---------
    TypeError

    Init_01
        mydict keys ar not correctly defined

    Init_02
        Specified measures column name is not found

    Init_03
        Dataframe contains null values

    Init_04
        Specified measures column a non numerical type column
    """

    def __init__(self, mydf, mydict):  # noqa
        """Initializate a new instance of a capability model"""
        # check dict is correctly defined
        lst_key = ["value", "batch", "LSL", "HSL", "goal"]
        if not all(key in mydict for key in lst_key):
            raise ValueError(
                f"Error init_01: wrong arg dictionary keys: {mydict.keys()} |"
                f" Be sure to use: {lst_key}"
            )
        lst_bol = [
            isinstance(mydict["LSL"], int),
            isinstance(mydict["LSL"], float),
            isinstance(mydict["HSL"], int),
            isinstance(mydict["HSL"], float),
        ]

        if True not in lst_bol:
            raise ValueError(
                f"Error init_05: neither LSL or HSL are numeric."
                f"LSL: {mydict['LSL']}"
                f" ,HSL: {mydict['HSL']} |"
                f" One must be numeric"
            )

        lst_goal = [
            isinstance(mydict["goal"], int),
            isinstance(mydict["goal"], float),
        ]
        if True in lst_goal:
            self.__goal = True
        else:
            self.__goal = False

        # Initializate different variables
        self.__log = list()
        df_cp = pd.DataFrame()

        # Save which tolerance is specified
        self.__Tol = 0  # None
        if lst_bol[0] is True or lst_bol[1] is True:
            self.__log.append("LSL is specified")
            self.__Tol = 1  # LSL is specified

        if True in [lst_bol[2], lst_bol[3]] and self.__Tol == 1:
            self.__log.append("HSL is specified")
            self.__Tol = 3  # HSL and LSL are specified
        elif True in [lst_bol[2], lst_bol[3]]:
            self.__log.append("HSL is specified")
            self.__Tol = 2  # HSL is specified

        # check measures data column is correct defined
        if mydict["value"] not in mydf.keys().tolist():
            raise ValueError(
                f"Error init_02: Specified dict_value: {mydict['value']} |"
                f" is not in the original dataframe column names:"
                f" {mydf.keys().tolist()}, please, review the class argument"
            )
        df_cp["Value"] = mydf[mydict["value"]]

        # Check if null values exist
        if df_cp.isna().sum().sum() > 0:
            raise ValueError(
                "Error init_03: Given dataframe contains"
                f" {df_cp.isna().sum().sum()} null values. Please, clean"
                " and filter your dataset before to call this method"
            )

        # Check if measures is a numeric column
        if pd.api.types.is_numeric_dtype(df_cp["Value"]) is False:
            raise ValueError(
                f"Error init_04: Given dataframe column {mydict['value']} must"
                " be numeric type. Please, review your dataset"
            )

        # In docs, remember mark the importance to read docs if long term
        # analysis is not done
        # Check if batch is defined. If so, stablish as categorical data type
        if mydict["batch"] not in mydf.keys().tolist():
            self.__log.append(
                f"{mydict['batch']} not match with column names"
                f" {mydf.keys().tolist()}"
            )
            self.__log.append("Short term analysis is activated")
            self.__AnalysisType = 1
        else:
            self.__log.append("Long term analysis is activated")
            self.__AnalysisType = 2
            df_cp["Batch"] = mydf[mydict["batch"]].astype("category")

        self.__df_cp = df_cp
        self.__log.append("Model is created")
        self.__mydict = mydict

    def getLog(self):
        """Return a string which contain all important calculations.

        Returns:
        ---------
        log: String
            all step logged
        """
        # Build up log string to be printed
        str_log = ""
        for i in range(0, len(self.__log)):
            str_log = str_log + self.__log[i] + "\n"

        return str_log

    def Normality_test(self):
        """Normality_test report is a figure that contain different chart and
        data description which helps to conclude if the measurement could be
        explained as a normal distribution so capability analysis could be
        done, or, inthe other hand, this analysis could not be take place.
        First, an histogram is showed, then, the probability plot. Third one
        is a time series plot and a descriptive data box.
        Finally, an advisement is showed based on p value.

        Returns:
        ---------
        Fig_NT : matplotlib figure
            Set of charts
        """
        df_work = self.__df_cp.copy()
        Fig_NT = plt.figure(figsize=(18, 12))
        Fig_NT.set_facecolor("white")
        gs = mpl.gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        Fig_NT.suptitle("\nNormality Test", fontsize=20)

        # ============================================================================================
        #                                Histogram
        # ============================================================================================
        ax1 = Fig_NT.add_subplot(gs[0, 0])
        ax1.set_title("Histogram")
        myweights = (
            np.ones(len(df_work["Value"])) / len(df_work["Value"]) * 100
        )
        ax1.hist(df_work["Value"], bins=100, weights=myweights)
        ax1.set_ylabel("% Weight")
        ax1.set_xlabel("Variable")

        # ============================================================================================
        #                                Probability Plot
        # ============================================================================================
        ax2 = Fig_NT.add_subplot(gs[0, 1])

        measurement = df_work["Value"].dropna().to_numpy()
        stat, p = stats.shapiro(measurement)
        ax2 = stats.probplot(measurement, dist="norm", plot=pylab)  # noqa

        # ============================================================================================
        #                                Time series
        # ============================================================================================
        ax3 = Fig_NT.add_subplot(gs[1, 0])
        ax3.set_title("Time series")
        ax3.plot(df_work.index, df_work["Value"], marker=".")
        ax3.set_ylabel("Variable")
        ax3.set_xlabel("Time")

        # ============================================================================================
        #                                Descriptive
        # ============================================================================================
        ax4 = Fig_NT.add_subplot(gs[1, 1])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_facecolor("white")
        ax4.set_ylim(0, 1)
        ax4.set_xlim(0, 1)
        ax4.annotate(
            "DESCRIPTIVE STATISTIC\n\n{}\n\np value: {:.2f}".format(
                df_work.describe(), p
            ),
            xy=(0.3, 0.2),
            bbox=dict(boxstyle="round", fc="w", color="black"),
            fontsize=12,
        )
        if p > 0.05:
            ax4.annotate(
                "Process seems to match with Normal distribution",
                xy=(0.1, 0.05),
                bbox=dict(boxstyle="round", fc="mediumseagreen"),
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax4.annotate(
                "Process seems to NOT match with Normal distribution",
                xy=(0.05, 0.05),
                bbox=dict(boxstyle="round", fc="red"),
                fontsize=12,
                fontweight="bold",
            )

        return Fig_NT

    def __Cp_Report(self):
        """Cp draws short term report

        Returns:
        ---------
        Fig_Cp : matplotlib figure
            Short Term Report
        """
        # Determine avg and standard deviation
        df_cp = self.__df_cp
        dbl_Total_avg = self.__df_cp["Value"].mean()
        SD = statistics.stdev(df_cp["Value"])

        # Solve Capability parameters
        if self.__Tol == 3:
            Cp = round(
                (self.__mydict["HSL"] - self.__mydict["LSL"]) / (6 * SD), 2
            )
            CpL = round((dbl_Total_avg - self.__mydict["LSL"]) / (3 * SD), 2)
            CpU = round((self.__mydict["HSL"] - dbl_Total_avg) / (3 * SD), 2)
            Cpk = min(CpL, CpU)
        elif self.__Tol == 1:
            Cp = "*"
            CpL = round((dbl_Total_avg - self.__mydict["LSL"]) / (3 * SD), 2)
            CpU = "*"
            Cpk = CpL
        elif self.__Tol == 2:
            Cp = "*"
            CpL = "*"
            CpU = round((self.__mydict["HSL"] - dbl_Total_avg) / (3 * SD), 2)
            Cpk = CpU

        # build up the model
        dbl_step = (df_cp["Value"].max() - df_cp["Value"].min()) * 0.01
        np_values = np.arange(
            df_cp["Value"].min(), df_cp["Value"].max(), dbl_step
        )

        Fig_Cp = plt.figure(figsize=(18, 12))
        Fig_Cp.set_facecolor("white")
        gs = mpl.gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
        Fig_Cp.suptitle(
            "\nProcess capability report for {}".format(
                self.__mydict["value"]
            ),
            fontsize=20,
        )

        # ============================================================================================
        #                                DATA
        # ============================================================================================
        ax1 = Fig_Cp.add_subplot(gs[0, 0])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_facecolor("white")
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 1)
        str_msg = "    Data Processing\n\nLSL                  {}\nGoal"
        str_msg = (
            str_msg + "{}\nHSL                 {}\nMean              {:.4f}"
        )
        str_msg = str_msg + "\nSample Size    {}\nStd. Dev (ST)  {:.4f}"
        ax1.annotate(
            str_msg.format(
                self.__mydict["LSL"],
                self.__mydict["goal"],
                self.__mydict["HSL"],
                dbl_Total_avg,
                len(df_cp.index),
                SD,
            ),
            xy=(0.3, 0.5),
            bbox=dict(boxstyle="round", fc="w", color="black"),
            fontsize=12,
        )

        # ============================================================================================
        #                                HISTOGRAM
        # ============================================================================================
        ax2 = Fig_Cp.add_subplot(gs[0, 1:3])
        ax2.hist(df_cp["Value"], bins=100, label="Data")
        ax2.set_yticks([])

        if self.__Tol == 3 or self.__Tol == 1:
            ax2.axvline(
                self.__mydict["LSL"], label="LSL", color="red", linestyle="--"
            )

        if self.__Tol == 3 or self.__Tol == 2:
            ax2.axvline(
                self.__mydict["HSL"], label="HSL", color="red", linestyle="--"
            )

        if self.__goal is True:
            ax2.axvline(
                self.__mydict["goal"],
                label="Goal",
                color="green",
                linestyle="--",
            )

        ax2.plot(
            np_values,
            stats.norm.pdf(np_values, dbl_Total_avg, SD),
            label="Short Term",
            color="grey",
            linestyle="--",
        )
        ax2.legend(loc="upper right")

        # ============================================================================================
        #                                Parameter
        # ============================================================================================
        ax3 = Fig_Cp.add_subplot(gs[0, 3])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_facecolor("white")
        ax3.set_ylim(0, 1)
        ax3.set_xlim(0, 1)
        str_msg = "ST Capability\n\n  Cp     {}\n  CpL   {}\n  CpU   {}"
        str_msg = str_msg + "\n  Cpk   {}\n"
        ax3.annotate(
            str_msg.format(Cp, CpL, CpU, Cpk),
            xy=(0.3, 0.4),
            bbox=dict(boxstyle="round", fc="w", color="black"),
            fontsize=12,
        )

        if Cpk > 1:
            str_adv = "\n   Process is capable"
            if self.__Tol == 3 and CpU > CpL:
                str_adv = str_adv + "\nand moved to the left"
            elif self.__Tol == 3 and CpU < CpL:
                str_adv = str_adv + "\nand moved to the rigth"

            str_adv = str_adv + f"\n\n     Cpk: {Cpk}>1\n"
            str_color = "mediumseagreen"
        else:
            str_adv = f"\nProcess is not capable\n\n     Cpk: {Cpk}<1\n"
            str_color = "red"

        ax3.annotate(
            str_adv,
            xy=(0.2, 0),
            bbox=dict(boxstyle="round", fc=str_color, color="black"),
            fontsize=12,
        )

        return Fig_Cp

    def __Pp_Report(self):  # noqa
        """Pp draws long term report

        Returns:
        ---------
        Fig_Lp : matplotlib figure
            long Term Report
        """
        # Determine avg and standard deviation for short and long term
        dbl_Param = 0
        dbl_SampleSize = 0
        dbl_iSampleSize = 0
        df_cp = self.__df_cp
        for item in df_cp["Batch"].unique():
            df_temp = df_cp[df_cp["Batch"] == item]
            dbl_iSampleSize = len(df_temp["Value"])
            dbl_SampleSize = dbl_SampleSize + dbl_iSampleSize
            dbl_Param = dbl_Param + (
                statistics.stdev(df_temp["Value"]) ** 2
            ) * (dbl_iSampleSize - 1)

        dbl_Total_avg = df_cp["Value"].mean()
        SDp = (
            dbl_Param / (dbl_SampleSize - len(df_cp["Batch"].unique()))
        ) ** (1 / 2)
        SD = statistics.stdev(df_cp["Value"])

        # Solve Capability parameters
        if self.__Tol == 3:
            Pp = round(
                (self.__mydict["HSL"] - self.__mydict["LSL"]) / (6 * SD), 2
            )
            PpL = round((dbl_Total_avg - self.__mydict["LSL"]) / (3 * SD), 2)
            PpU = round((self.__mydict["HSL"] - dbl_Total_avg) / (3 * SD), 2)
            Ppk = min(PpL, PpU)

            Cp = round(
                (self.__mydict["HSL"] - self.__mydict["LSL"]) / (6 * SDp), 2
            )
            CpL = round((dbl_Total_avg - self.__mydict["LSL"]) / (3 * SDp), 2)
            CpU = round((self.__mydict["HSL"] - dbl_Total_avg) / (3 * SDp), 2)
            Cpk = min(CpL, CpU)

        elif self.__Tol == 1:
            Pp = "*"
            PpL = round((dbl_Total_avg - self.__mydict["LSL"]) / (3 * SD), 2)
            PpU = "*"
            Ppk = PpL
            Cp = "*"
            CpL = round((dbl_Total_avg - self.__mydict["LSL"]) / (3 * SDp), 2)
            CpU = "*"
            Cpk = CpL
        elif self.__Tol == 2:
            Pp = "*"
            PpL = "*"
            PpU = round((self.__mydict["HSL"] - dbl_Total_avg) / (3 * SD), 2)
            Ppk = PpU
            Cp = "*"
            CpL = "*"
            CpU = round((self.__mydict["HSL"] - dbl_Total_avg) / (3 * SDp), 2)
            Cpk = CpU

        dbl_step = (df_cp["Value"].max() - df_cp["Value"].min()) * 0.01
        np_values = np.arange(
            df_cp["Value"].min(), df_cp["Value"].max(), dbl_step
        )

        Fig_Lp = plt.figure(figsize=(18, 12))
        Fig_Lp.set_facecolor("white")
        gs = mpl.gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
        Fig_Lp.suptitle(
            "\nProcess capability report for {}".format(
                self.__mydict["value"]
            ),
            fontsize=20,
        )

        # ============================================================================================
        #                                DATA
        # ============================================================================================
        ax1 = Fig_Lp.add_subplot(gs[0, 0])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_facecolor("white")
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 1)
        str_msg = (
            "    Data Processing\n\nLSL                  {}\nGoal"
            + "                {}"
            + "\nHLS                 {}\nMean              {:.4f}\nSample Size"
        )
        str_msg = str_msg + "{}\nStd. Dev (LT)   {:.4f}\nStd. Dev (ST)  {:.4f}"
        ax1.annotate(
            str_msg.format(
                self.__mydict["LSL"],
                self.__mydict["goal"],
                self.__mydict["HSL"],
                dbl_Total_avg,
                len(df_cp.index),
                SD,
                SDp,
            ),
            xy=(0.3, 0.2),
            bbox=dict(boxstyle="round", fc="w", color="black"),
            fontsize=12,
        )

        # ============================================================================================
        #                                HISTOGRAM
        # ============================================================================================
        ax2 = Fig_Lp.add_subplot(gs[0, 1:3])
        ax2.hist(df_cp["Value"], bins=100, label="Data")
        ax2.set_yticks([])

        if self.__Tol == 3 or self.__Tol == 1:
            ax2.axvline(
                self.__mydict["LSL"], label="LSL", color="red", linestyle="--"
            )

        if self.__Tol == 3 or self.__Tol == 2:
            ax2.axvline(
                self.__mydict["HSL"], label="HSL", color="red", linestyle="--"
            )

        if self.__goal is True:
            ax2.axvline(
                self.__mydict["goal"],
                label="Goal",
                color="green",
                linestyle="--",
            )

        ax2.plot(
            np_values,
            stats.norm.pdf(np_values, dbl_Total_avg, SD),
            label="LT",
            color="black",
        )
        ax2.plot(
            np_values,
            stats.norm.pdf(np_values, dbl_Total_avg, SDp),
            label="ST",
            color="grey",
            linestyle="--",
        )
        ax2.legend(loc="upper right")

        # ============================================================================================
        #                                Parameter
        # ============================================================================================
        ax3 = Fig_Lp.add_subplot(gs[0, 3])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_facecolor("white")
        ax3.set_ylim(0, 1)
        ax3.set_xlim(0, 1)
        str_msg = (
            " LT Cpacility \n\n  Pp     {} \n"
            + "  PpL   {} \n  PpU   {} \n  Ppk   {} \n"
        )
        ax3.annotate(
            str_msg.format(Pp, PpL, PpU, Ppk),
            xy=(0.13, 0.4),
            bbox=dict(boxstyle="round", fc="w", color="black"),
            fontsize=12,
        )
        str_msg = (
            " ST Cpacility \n\n  Cp     {} \n  CpL"
            + "   {} \n  CpU   {} \n  Cpk   {} \n"
        )
        ax3.annotate(
            str_msg.format(Cp, CpL, CpU, Cpk),
            xy=(0.55, 0.4),
            bbox=dict(boxstyle="round", fc="w", color="black"),
            fontsize=12,
        )

        if Cpk > 1:
            str_adv = "\n   Process is capable"
            if self.__Tol == 3 and CpU > CpL:
                str_adv = str_adv + "\nand moved to the left"
            elif self.__Tol == 3 and CpU < CpL:
                str_adv = str_adv + "\nand moved to the rigth"

            str_adv = str_adv + f"\n\n     Cpk: {Cpk}>1\n"
            str_color = "mediumseagreen"
        else:
            str_adv = f"\nProcess is not capable\n\n     Cpk: {Cpk}<1\n"
            str_color = "red"

        ax3.annotate(
            str_adv,
            xy=(0.2, 0),
            bbox=dict(boxstyle="round", fc=str_color, color="black"),
            fontsize=12,
        )

        return Fig_Lp

    def Report(self):
        """Report is a figure that contain a chart where measures distirbution
        is showed with its tolerance. Then capability factor is determined
        and showed

        Returns:
        ---------
        myFigure : matplotlib figure
            Set of charts
        """
        if self.__AnalysisType == 1:
            myFigure = self.__Cp_Report()
        elif self.__AnalysisType == 2:
            myFigure = self.__Pp_Report()
        else:
            raise ValueError("Error Rep_01: Cp_Model is not defined correctly")

        return myFigure
