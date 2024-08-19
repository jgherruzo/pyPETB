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
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypetb import tables

warnings.simplefilter(action="ignore", category=FutureWarning)


class RNumeric:
    """Repeatability numeric gage analysis.
    RNumeric works as a model. It is defined using a measurement dataframe.
    Log method could be called in order to check each parameter calculation.
    There are possibilities to get an anova table, standard deviation table or
    variance table, which are returned as pandas dataframe, or Report, where a
    matplotlib figure with the full analysis will be returned.
    Be sure to import matplotlib.pyplot in your script and write plt.show()
    after call .R_Report() or .Run_chart() to get the figures. Importind
    seaborn as sns and sns.set() is highly recommended to improve the sigth of
    the reports.

    Args:
    -------
    mydf_Raw : Pandas DataFrame containing 2 columns
        part and value

    mydict_key : Dictionary containing column names.
        key 1 for Part column
        , key 2 for Value column

    mydbl_tol : Optional. Int or Float
        system tolerance

    Methods:
    ----------
    getLog: string
          printable string containing all individual calculations

    RSolve: none
          determine all variables to make the RnR analysis

    RAnova: Pandas DataFrame
          two way anova

    R_varTable: Pandas DataFrame
          RnR variance analysis

    R_SDTable: Pandas DataFrame
          RnR standard deviation analysis

    R_RunChart: Figure
          Running chart for your measurement

    R_Report: Figure
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

    Sstd2_: float
          Sum of deviation by part

    SSpart: float
          Total Part Sum of deviation

    SStotal: float
          Total squared deviation

    SSequipment: float
          Equipment squared deviation

    ndc: int
          number of distinct categories

    Raises:
    ---------
    TypeError

    Init_01
        When input mydict_key contains a column which is not a mydf_Raw column
        name

    Init_02
        mydf_Raw contain nan values

    Init_03
        mydict_key['2'] is a non numerical type column

    Init_04
        mydbl_tol is not a number

    Init_05
        mydict_key is not correctly defined
    """

    def __init__(self, mydf_Raw, mydict_key, mydbl_tol=None):
        """Initializate a new instance of a numeric R model"""
        self.__dict_key = mydict_key
        self.__dbl_tol = mydbl_tol

        # Check dictionary is correctly defined
        lst_key = ["1", "2"]
        if not all(key in mydict_key for key in lst_key):
            raise ValueError(
                f"Error init_01: wrong dictionary keys: {mydict_key.keys()} |"
                f" Be sure to use: {lst_key}"
            )

        if mydict_key["1"] not in mydf_Raw.keys().tolist():
            bol_key = True
            str_Column = mydict_key["1"]
            int_key = 1
        elif mydict_key["2"] not in mydf_Raw.keys().tolist():
            bol_key = True
            str_Column = mydict_key["2"]
            int_key = 2
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

        # Check if there is any null number
        if mydf_Raw.isna().sum().sum() > 0:
            raise ValueError(
                "Error init_02: Given dataframe contains"
                f" {mydf_Raw.isna().sum().sum()} null values. Please, clean"
                " and filter your dataset before to call this method"
            )

        if pd.api.types.is_numeric_dtype(mydf_Raw[mydict_key["2"]]) is False:
            raise ValueError(
                f"Error init_03: Given dataframe column {mydict_key['2']} must"
                " be numeric type. Please, review your dataset"
            )

        # Create main working dataframe
        df_0 = pd.DataFrame()
        df_0["Part"] = mydf_Raw[mydict_key["1"]]
        df_0["Valor"] = mydf_Raw[mydict_key["2"]]
        df_0["Part"] = df_0["Part"].astype("category")
        df_0["Valor"] = pd.to_numeric(df_0["Valor"])
        df_0.sort_values(by=["Part"], inplace=True)

        self.__log = list()
        self.__df_0 = df_0
        self.__log.append("Model is created")
        self.__Status = 1

    def getLog(self):
        """Return a string which contain all important calculations.

        Returns:
        ---------
        log: String
            all step logged
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")

        str_log = ""
        for i in range(0, len(self.__log)):
            str_log = str_log + self.__log[i] + "\n"

        return str_log

    def RSolve(self, bol_bias=False):
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
            raise ValueError("You need at least one instance of RNumeric")

        mydf_0 = self.__df_0

        if bol_bias is False:
            if (
                len(
                    mydf_0.groupby("Part", observed=False)
                    .count()["Valor"]
                    .unique()
                    .tolist()
                )
                > 1
            ):
                raise ValueError(
                    """Error Solve_01: Some pieze has different
                                  number of runs. If would like to proceed,
                                  please, call .RSolve(bol_bias=True)"""
                )

        # One column per run
        df_temp2 = mydf_0[mydf_0["Part"] == mydf_0["Part"].unique()[0]]

        # number of Piezes
        self.t = 1
        self.p = len(mydf_0["Part"].unique())
        self.n_parts = len(mydf_0["Part"].unique())

        # number of runs
        self.r = len(df_temp2)
        lst_columns = ["Run " + str(i) for i in range(0, self.r)]

        self.__log.append(
            "== DATASET EVALUATION ==\nTrials: {}\nPiezes: {}".format(  # noqa
                self.r, self.p
            )
        )

        # a dataframe containing a run per column is generated
        df = pd.DataFrame(columns=["Part"] + lst_columns)

        df_temp1 = mydf_0
        for part in df_temp1["Part"].unique():
            df_temp2 = df_temp1[df_temp1["Part"] == part]
            dict_value = dict()
            dict_value["Part"] = part
            i = 0
            for value in range(0, len(df_temp2["Valor"])):
                test = "Run " + str(i)
                dict_value[test] = df_temp2["Valor"].iloc[value]
                i += 1

            df_dictionary = pd.DataFrame([dict_value])
            df = pd.concat([df, df_dictionary], ignore_index=True)

        df.set_index(["Part"], inplace=True)

        df_1 = df.copy()
        df_1["Range"] = df.max(axis=1) - df.min(axis=1)
        df_1["Mean"] = df.mean(axis=1)

        # Create a dataframe gouped by part for
        # future calculations
        df_3 = df_1.groupby("Part").mean()

        # Start making calculations
        self.__log.append("== CALCULATION ==")
        self.Total_Data = self.r * self.p
        self.__log.append("Total data: %.0f" % self.Total_Data)

        self.Total_max = np.max(df.max())
        self.__log.append("Max. measured value: %.4f" % self.Total_max)

        self.Total_min = np.min(df.min())
        self.__log.append("Min. measured value: %.4f" % self.Total_min)

        tbl = tables.Stat_Tables()
        # Determine averange range, Range UCL and LCL
        n_t = 2  # minimum required
        self.dbl_Range_avg = df_1["Range"].mean()
        self.dbl_Range_UCL = self.dbl_Range_avg * tbl.get_D4(n_t)
        self.dbl_Range_LCL = self.dbl_Range_avg * tbl.get_D3(n_t)

        # Determine whole average, UCL and LCL
        self.Total_avg = np.mean(df.mean())
        self.__log.append("Avg. measured value: %.4f" % self.Total_avg)
        self.dbl_Avg_UCL = (
            self.Total_avg + tbl.get_A2(n_t) * self.dbl_Range_avg
        )
        self.dbl_Avg_LCL = (
            self.Total_avg - tbl.get_A2(n_t) * self.dbl_Range_avg
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

        df_3["std2"] = (df_3["Mean"] - self.Total_avg) ** 2

        self.Sstd2_ = df_3["std2"].sum()
        self.__log.append(
            "Sum of deviation by part: {:.6f}".format(self.Sstd2_)
        )

        self.SSpart = self.r * self.Sstd2_
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

        self.__df = df

        self.__df_3 = df_3
        self.__df_1 = df_1
        # print(df_1)
        self.__Status = 2

    def RAnova(self):
        """After calling .RSolve() anova available calculations could be done.
        It will be returned as pandas DataFrame and all of the values will be
        accesibles from the dataframe.

        Returns:
        ---------
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
            "Part",
            "Repeatability with",
            "Repeatability without",
            "Total",
        ]
        df_Anova["DF"] = [
            (self.p - 1),
            ((self.t * self.p) * (self.r - 1)),
            ((self.t * self.p) * (self.r - 1) + (self.t - 1) * (self.p - 1)),
            (self.t * self.p * self.r - 1),
        ]

        df_Anova["SS"] = [
            (self.SSpart),
            (self.SSequipment),
            (self.SSequipment + 0),
            (self.SStotal),
        ]

        MSpart = self.SSpart / (self.p - 1)
        MSequipment = self.SSequipment / (self.t * self.p * (self.r - 1))

        df_Anova["MS"] = [
            (MSpart),
            (MSequipment),
            (self.SSequipment + 0)
            / ((self.t * self.p) * (self.r - 1) + (self.t - 1) * (self.p - 1)),
            (np.nan),
        ]

        df_Anova.set_index("Source of variability", inplace=True)
        return df_Anova

    def R_varTable(self):
        """After calling .RSolve() variance table could be done.
        It will be returned as pandas DataFrame and all of the values will be
        accesibles from the dataframe.

        Returns:
        ---------
        Pandas DataFrame
            Variante table result tabulated into a pandas dataframe
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")
        elif self.__Status == 1:
            raise ValueError(
                "You must call RnR.RnRSolve() before to call this method"
            )

        df_Anova = self.RAnova()
        Srpeatability = self.SSequipment / (self.t * self.p * (self.r - 1))

        MSpart = df_Anova.loc["Part"]["MS"]
        Spart = (MSpart) / (self.r * self.t)

        GRnR = Srpeatability
        PtP = Spart
        TV = Srpeatability + PtP
        df_varTbl = pd.DataFrame()
        df_varTbl["Source"] = [
            "Gage Repeatability",
            "Part to Part",
            "Total variation",
        ]
        df_varTbl["Variance"] = [
            (GRnR),
            (PtP),
            (TV),
        ]
        df_varTbl["% Contribution"] = [
            (GRnR / TV * 100),
            (PtP / TV * 100),
            (TV / TV * 100),
        ]
        df_varTbl.set_index("Source", inplace=True)
        return df_varTbl

    def R_SDTable(self):
        """After calling .RSolve() standard deviation table could be done.

        It
        will be returned as pandas DataFrame and all of the values will be
        accesibles from the dataframe.

        Returns:
        ---------
        Pandas DataFrame
            Standard deviation table result tabulated into a pandas dataframe
        """
        if self.__Status is None:
            raise ValueError("You need at least one instance")
        elif self.__Status == 1:
            raise ValueError(
                "You must call RnR.RnRSolve() before to call this method"
            )
        df_varTbl = self.R_varTable()
        GRnR = df_varTbl.loc["Gage Repeatability"]["Variance"]
        PtP = df_varTbl.loc["Part to Part"]["Variance"]
        TV = df_varTbl.loc["Total variation"]["Variance"]

        df_SDTbl = pd.DataFrame()
        df_SDTbl["Source"] = [
            "Gage Repeatability",
            "Part to Part",
            "Total variation",
        ]
        df_SDTbl["StdDev (SD)"] = [
            (GRnR ** (1 / 2)),
            (PtP ** (0.5)),
            (TV ** (0.5)),
        ]
        df_SDTbl["StudyVar (6*SD)"] = [
            (6 * GRnR ** (1 / 2)),
            (6 * PtP ** (0.5)),
            (6 * TV ** (0.5)),
        ]
        df_SDTbl["% Study Var"] = [
            (GRnR ** (1 / 2) / TV ** (0.5) * 100),
            (PtP ** (0.5) / TV ** (0.5) * 100),
            (TV ** (0.5) / TV ** (0.5) * 100),
        ]
        df_SDTbl.set_index("Source", inplace=True)

        if self.__dbl_tol is not None:
            df_SDTbl["% tol (VE/tol)"] = [
                (6 * GRnR ** (1 / 2)) / self.__dbl_tol * 100,
                (6 * PtP ** (0.5)) / self.__dbl_tol * 100,
                (6 * TV ** (0.5)) / self.__dbl_tol * 100,
            ]

        self.ndc = (
            np.sqrt(2)
            * df_SDTbl["StdDev (SD)"].loc["Part to Part"]
            / df_SDTbl["StdDev (SD)"].loc["Gage Repeatability"]
        )

        return df_SDTbl

    def R_RunChart(self):
        """Run chart is a figure that contain a chart per pieze where all the
        measurement made by the operator are showed.

        Returns:
        ---------
        Fig1 : matplotlib figure
            Set of charts
        """
        Fig1 = plt.figure(figsize=(18, 12))
        Fig1.set_facecolor("white")
        gs = mpl.gridspec.GridSpec(
            int((self.p / 5) + 1), 5, wspace=0, hspace=0.1
        )
        Fig1.suptitle(
            "\nRUN CHART FOR MEASUREMENT SYSTEM OF {} BY {}".format(
                self.__dict_key["2"], self.__dict_key["1"]
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
            "CHART CALIBRATION\nMax. {:.3f}\nAvg. {:.3f}\nMin. {:.3f}".format(
                self.Total_max, self.Total_avg, self.Total_min
            ),
            xy=(0.1, 0.2),
            bbox=dict(boxstyle="round", fc="w", color="lightgrey"),
            fontsize=12,
        )

        lst_TT[1].annotate(
            "Panel var: {}".format(self.__dict_key["1"]),
            xy=(0, 0.1),
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=12,
            fontstyle="italic",
        )

        # initializate values for char drawing
        row = 1
        column = 0
        lst_ax = list()

        # Chose operator parameters
        dict_OperatorLine = dict()
        str_op_key = "1"
        dict_OperatorLine[str_op_key] = {
            "Color": random.choice(list(mpl.colors.CSS4_COLORS.values())),
            "Marker": random.choice(list(mpl.lines.Line2D.markers.keys())),
        }
        # print(dict_OperatorLine)

        # Create run chart
        for counter in range(0, self.p):
            # One box per pieze
            str_piece = self.__df.index.unique(level="Part")[counter]
            # df_temp = self.__df.xs(str_piece, level=1, drop_level=False)
            df_temp = self.__df.loc[self.__df.index == str_piece]

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
            # per pieze one blocks
            t_base = 0
            x = np.arange(t_base, t_base + len(range(0, self.r)))

            str_op = str_op_key
            y = df_temp.values.flatten()

            lst_ax[counter].plot(
                x,
                y,
                label=str_op,
                color=dict_OperatorLine[str_op_key]["Color"],
                marker=dict_OperatorLine[str_op_key]["Marker"],
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

    def R_Report(self, report_name=None):  # noqa: C901
        """R_Report chart is a figure that contain six important chart in
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

        Returns:
        ---------
        Fig2 : matplotlib figure
            Set of charts
        """
        # df = self.__df
        df_0 = self.__df_0
        df_3 = self.__df_3
        # df_2 = self.__df_2
        df_1 = self.__df_1

        Fig2 = plt.figure(figsize=(16, 12))
        Fig2.set_facecolor("white")

        ncols = 2
        # Each graph gets (nrows - 1) / 3 rows each.
        # Last row is used for the 'Final Thoughts' section.
        # Set total number of rows high enough to optimize the height
        # of the final section to minimize the waste of vertical space
        nrows = 22

        # Variables used to indicate where in the GridSpec created below
        # each of the graphs should be placed.
        first_row = slice(0, int(nrows / 3) - 1)
        second_row = slice(int(nrows / 3) + 1, (int(nrows / 3) * 2))
        third_row = slice((int(nrows / 3) * 2) + 2, nrows)

        first_col = 0
        second_col = 1

        gs = mpl.gridspec.GridSpec(
            nrows, ncols, wspace=0.5, hspace=0.9, figure=Fig2
        )
        default_title = "Repeatability Measurement System Report"
        title = report_name if report_name is not None else default_title

        Fig2.suptitle(
            title,
            fontsize=20,
        )

        technician_colors = [
            mpl.colors.CSS4_COLORS["blue"],
        ]

        technician_markers = [
            "o",  # Circle
        ]

        # ============================================================================================
        #                                VARIACION
        # ============================================================================================
        df_varTbl = self.R_varTable()
        GRnR = df_varTbl.loc["Gage Repeatability"]["Variance"]
        PtP = df_varTbl.loc["Part to Part"]["Variance"]
        TV = df_varTbl.loc["Total variation"]["Variance"]
        # Stechnician = df_varTbl.loc["Technician"]["Variance"]
        # StechniciaxPart = df_varTbl
        #           .loc["Technician x Part iter."]["Variance"]

        x = ["Repea.", "PtP"]
        y = {
            "% Contribution": (
                GRnR / TV * 100,
                PtP / TV * 100,
            ),
            "% Var. Study": (
                GRnR ** (1 / 2) / TV ** (0.5) * 100,
                PtP ** (0.5) / TV ** (0.5) * 100,
            ),
        }

        if self.__dbl_tol is not None:
            y.update(
                {
                    "% tol (VE/tol)": (
                        (6 * GRnR ** (1 / 2)) / self.__dbl_tol * 100,
                        (6 * PtP ** (0.5)) / self.__dbl_tol * 100,
                    )
                }
            )

        # print(y)

        ax1 = Fig2.add_subplot(gs[first_row, first_col])
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
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax1.set_ylim(0, 100)

        # ============================================================================================
        #                                Value per Piece
        # ============================================================================================
        ax2 = Fig2.add_subplot(gs[first_row, second_col])
        ax2.set_title(
            "{} by {}".format(self.__dict_key["2"], self.__dict_key["1"]),
            fontweight="bold",
        )
        ax2.scatter(df_0["Part"], df_0["Valor"])
        ax2.plot(df_3.index, df_3["Mean"], color="orange")
        ax2.set_xlabel("{}".format(self.__dict_key["1"]))
        ax2.set_ylabel("{}".format(self.__dict_key["2"]))
        ax2.set_xticks(df_3.index)

        # ============================================================================================
        #                                R per Operator
        # ============================================================================================
        ax_max = max(df_1["Range"].max(), self.dbl_Range_UCL) * 1.1
        ax_min = min(df_1["Range"].min(), self.dbl_Range_UCL) - ax_max * 0.1

        ax3 = Fig2.add_subplot(gs[second_row, first_col])
        ax3.set_ylabel("Sample Range")
        ax3.set_title("System")
        ax3.set_ylim(ax_min, ax_max)
        ax3.axhline(
            self.dbl_Range_UCL,
            color="red",
            label="UCL={:.4f}".format(self.dbl_Range_UCL),
        )
        ax3.axhline(
            self.dbl_Range_avg,
            color="black",
            linestyle="--",
            label="avg={:.4f}".format(self.dbl_Range_avg),
        )
        ax3.axhline(
            self.dbl_Range_LCL,
            color="red",
            label="LCL={:.4f}".format(self.dbl_Range_LCL),
        )
        df_temp = df_1
        ax3.plot(
            df_temp.index,
            df_temp["Range"],
            color=technician_colors[0],
            marker=technician_markers[0],
        )
        str_label = self.__dict_key["1"]

        ax3.set_xlabel(str_label)  #
        ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # ============================================================================================
        #                                Violin Plot
        # ============================================================================================
        ax4 = Fig2.add_subplot(gs[second_row, second_col])
        ax4.set_title(
            "{} by {}".format(self.__dict_key["2"], self.__dict_key["1"]),
            fontweight="bold",
        )
        c = technician_colors[0]
        ax4.boxplot(
            df_0["Valor"],
            positions=[1],
            patch_artist=True,
            medianprops=dict(color=mpl.colors.CSS4_COLORS["black"]),
            boxprops=dict(facecolor=c, color=c),
        )
        ax4.set_ylabel("{}".format(self.__dict_key["2"]))
        ax4.set_xticklabels(["System"])
        # ============================================================================================
        #                                Xbarra per Operator
        # ============================================================================================
        ax5_max = self.Total_max * 1.001
        ax5_min = self.Total_min - self.Total_max * 0.001

        ax5 = Fig2.add_subplot(gs[third_row, first_col])
        ax5.set_ylabel("Sample Avg")
        ax5.set_title("System")
        ax5.set_ylim(ax5_min, ax5_max)

        ax5.axhline(
            self.dbl_Avg_UCL,
            color="red",
            label="UCL={:.4f}".format(self.dbl_Avg_UCL),
        )
        ax5.axhline(
            self.Total_avg,
            color="black",
            linestyle="--",
            label="avg={:.4f}".format(self.Total_avg),
        )
        ax5.axhline(
            self.dbl_Avg_LCL,
            color="red",
            label="LCL={:.4f}".format(self.dbl_Avg_LCL),
        )

        df_temp = df_1

        ax5.plot(
            df_temp.index,
            df_temp["Mean"],
            color=technician_colors[0],
            marker=technician_markers[0],
        )

        ax5.set_xlabel("{}".format(self.__dict_key["1"]))  #
        ax5.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # ============================================================================================
        #                                Iteration
        # ============================================================================================
        ax6 = Fig2.add_subplot(gs[third_row, second_col])
        ax6.set_title(
            "Iteration {}x{}".format(
                self.__dict_key["2"], self.__dict_key["1"]
            ),
            fontweight="bold",
        )
        df_temp = df_1
        ax6.plot(df_temp.index, df_temp["Mean"], label="System")

        ax6.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax6.set_xlabel("{}".format(self.__dict_key["1"]))
        ax6.set_ylabel("Mean")
        # ============================================================================================
        #                                Final Thoughts
        # ============================================================================================
        df = self.R_SDTable()
        dbl_RnR = df["% Study Var"].loc["Gage Repeatability"]
        dbl_ndc = self.ndc

        str_msg = f"Gage result: {dbl_RnR:.2f}% |"
        str_msg = str_msg + f" Number of distinc Categories: {dbl_ndc:.1f}\n\n"

        if dbl_RnR < 10 and dbl_ndc > 5:
            str_msg = str_msg + "The Measurement system seems to be OK"
            str_color = "mediumseagreen"
        elif dbl_RnR >= 10 and dbl_RnR <= 30 and dbl_ndc > 5:
            str_color = "yellow"
            str_msg = str_msg + (
                "The Measurement system may be acceptable depending on "
                + "application and cost\n\n"
                + "If want to improve, check your gage"
            )
        else:
            str_color = "red"
            str_msg = str_msg + "Unacceptable measurement system\n\n"
            if dbl_RnR > 30:
                str_msg = str_msg + "Check your gage\n\n"
            if dbl_ndc <= 5:
                str_msg = str_msg + "Your system has low accuracy (NDC<5)"

        self.final_thoughts = Fig2.text(
            0.5,
            0.01,
            str_msg,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round", facecolor=str_color, edgecolor="black"
            ),
        )

        plt.subplots_adjust(bottom=0.16)

        return Fig2
