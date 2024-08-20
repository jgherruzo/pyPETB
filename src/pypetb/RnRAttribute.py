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

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


class RnRAtribute:
    """Repeatability and Reproducibility analysis for atributes measurement.
    RnRAtribute works as a model. It is defined using a measurement dataframe.
    Once model is solved, RnR report could be extracted. If atributes
    measurement is yes/no, a yes/no matrix could be analyzed.

    Args:
    -------
    df_Raw : Pandas DataFrame containing 3 columns
        Operator, part and value

    mydict_key : Dictionary containing column names.
        key 1 to Operator column
        , key 2 for Part column
        , key 3 for expert/ref column
        , key 4 for measure value

    Methods:
    ----------
    getLog: string
          printable string containing all individual calculations

    Report: Figure
          RnR whole report

    Raises:
    ---------
    TypeError

    Init_01
        Input mydict_key keys are not corretly defined

    Init_02
        Input mydict_key values are not contained on source
         dataframe
    """

    def __RnRSolve(self):
        """Purpose   : Make all the required calculations
          DateTime  : 11/11/2023
          Author    : Jose García Herruzo
          Updates   :
            DATE        USER    DESCRIPTION
            N/A
        @Arg:
            N/A
        @Get:
            N/A
        """
        df_0 = self.__df_0

        self.__r = (
            df_0.groupby(["Part", "Op"], observed=True)
            .count()["Valor"]
            .unique()[0]
        )

        self.__log.append(f"Number of trials per operator: {self.__r}")

        self.__p = len(df_0["Part"].unique())
        self.__log.append(f"Number of parts: {self.__p}")

        # Determine RnR per Operator
        dict_Op = dict()
        for operator in df_0["Op"].unique().tolist():
            df_temp = df_0[df_0["Op"] == operator]
            lon_Rep = 0
            lon_Acc = 0
            for part in df_temp["Part"].unique().tolist():
                df_temp_v2 = df_temp[df_temp["Part"] == part]
                if len(df_temp_v2["Valor"].unique()) == 1:
                    lon_Rep = lon_Rep + 1
                    if (
                        df_temp_v2.iloc[0]["Valor"]
                        == df_temp_v2.iloc[0]["Ref"]
                    ):
                        lon_Acc = lon_Acc + 1

            dict_Op[operator] = {
                "Rep": lon_Rep / self.__p * 100,
                "Acc": lon_Acc / self.__p * 100,
            }
            self.__log.append(
                f"{operator} repeatability: {dict_Op[operator]['Rep']:.2f}%"
            )
            self.__log.append(
                f"{operator} accuracy: {dict_Op[operator]['Acc']:.2f}%"
            )

        # Determine Total values based on atributes concordance
        lon_Rep = 0
        lon_Acc = 0
        for part in df_0["Part"].unique().tolist():
            df_temp = df_0[df_0["Part"] == part]

            if len(df_temp["Valor"].unique()) == 1:
                lon_Rep = lon_Rep + 1
                if df_temp_v2.iloc[0]["Valor"] == df_temp_v2.iloc[0]["Ref"]:
                    lon_Acc = lon_Acc + 1

        dict_Op["System"] = {
            "Rep": lon_Rep / self.__p * 100,
            "Acc": lon_Acc / self.__p * 100,
        }
        str_temp = dict_Op["System"]["Rep"]
        self.__log.append(
            f"Total repeatability (Concordance of atributes): {str_temp:.2f}%"
        )
        str_temp = dict_Op["System"]["Acc"]
        self.__log.append(
            f"Total accuracy (Concordance of atributes): {str_temp:.2f}%"
        )

        self.__dict_Op = dict_Op

        df_0["Acc"] = df_0["Ref"] == df_0["Valor"]
        df_0["Acc"] = df_0["Acc"].astype(bool)
        dict_Sys = dict()
        dict_Sys["System"] = (
            df_0["Acc"].value_counts()[True] / len(df_0.index) * 100
        )
        self.__log.append(f"System accuracy: {dict_Sys['System']:.2f}%")
        df_sum = df_0.groupby("Op", observed=True)["Acc"].sum()
        df_1 = df_sum / self.__p / self.__r * 100
        df_1 = df_1.reset_index()
        df_1.set_index("Op", inplace=True)
        self.__df_1 = df_1

        for index, item in df_1.iterrows():
            dict_Sys[index] = item["Acc"]
            self.__log.append(f"{index} accuracy: {item['Acc']:.2f}%")

        self.__dict_Sys = dict_Sys

    def __init__(self, mydf_Raw, mydict_key):
        """Initializate a new instance of a numeric RnR model"""

        self.__dict_key = mydict_key
        # Check dictionary is correctly defined
        lst_key = ["1", "2", "3", "4"]
        if not all(key in mydict_key for key in lst_key):
            raise ValueError(
                f"Error init_01: wrong dictionary keys: {mydict_key.keys()} |"
                f" Be sure to use: {lst_key}"
            )

        if not all(
            key in mydict_key.values() for key in mydf_Raw.columns.tolist()
        ):
            raise ValueError(
                f"Error init_02: wrong dictionary values:"
                f" {mydict_key.values()} |"
                f" Be sure to specify the correct columns"
                f": {mydf_Raw.columns.tolist()}"
            )

        self.__log = list()

        # Create main working dataframe
        df_0 = pd.DataFrame()
        df_0["Op"] = mydf_Raw[mydict_key["1"]]
        df_0["Part"] = mydf_Raw[mydict_key["2"]]
        df_0["Ref"] = mydf_Raw[mydict_key["3"]]
        df_0["Valor"] = mydf_Raw[mydict_key["4"]]
        df_0["Op"] = df_0["Op"].astype("category")
        df_0["Part"] = df_0["Part"].astype("category")
        df_0["Ref"] = df_0["Ref"].astype("category")
        df_0["Valor"] = df_0["Valor"].astype("category")

        self.__df_0 = df_0
        self.__RnRSolve()

        self.__log.append("Model is created")

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

    def Report(self):
        """Return a figure to deal with attributes RnR analysis.
        It shows two focus, one based on the attributes concordance,
        which is more restrictive and the other based on ref. values.

        Returns:
        ---------
        Fig_R : matplotlib figure
            Set of charts
        """
        Fig_R = plt.figure(figsize=(15, 15))
        Fig_R.set_facecolor("white")
        gs = mpl.gridspec.GridSpec(
            5, 4, wspace=0.4, hspace=0.2, height_ratios=[10, 6, 5, 5, 6]
        )
        Fig_R.suptitle("\nAttributes R&R Analysis", fontsize=20)

        df = pd.DataFrame(self.__dict_Op)
        df = df.T
        df.drop("System", axis=0, inplace=True)
        df.sort_index(inplace=True)
        # ==============================================
        #          Concordance Analysis
        # ==============================================
        ax1 = Fig_R.add_subplot(gs[0, 0:2])
        ax1.set_title(
            "Repeteability based on attributes concordance", fontweight="bold"
        )
        ax1.barh(df.index, df["Rep"])
        ax1.set_xlim(0, 100)
        ax1.axvline(
            self.__dict_Op["System"]["Rep"],
            color="red",
            linestyle="--",
            label=f"{self.__dict_Op['System']['Rep']:.1f}% | SYSTEM",
        )
        ax1.legend()

        ax2 = Fig_R.add_subplot(gs[0, 2:4])
        ax2.set_title(
            "Accuracy based on attributes concordance", fontweight="bold"
        )
        ax2.barh(df.index, df["Acc"])
        ax2.set_xlim(0, 100)
        ax2.axvline(
            self.__dict_Op["System"]["Acc"],
            color="red",
            linestyle="--",
            label=f"{self.__dict_Op['System']['Acc']:.1f}% | SYSTEM",
        )
        ax2.legend()

        ax3 = Fig_R.add_subplot(gs[1, :])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_facecolor("white")
        ax3.set_ylim(0, 1)
        ax3.set_xlim(0, 1)

        str_msg = (
            f"\n  Concordance analysis:\n"
            f"      Repeteability    | "
            f"{self.__dict_Op['System']['Rep']:.1f}%  \n"
            f"      Reproducibility  | "
            f"{self.__dict_Op['System']['Acc']:.1f}%  "
        )

        if self.__dict_Op["System"]["Acc"] > 80:
            str_msg = str_msg + "\n\n   SYSTEM ACCEPTABLE\n"
            str_color = "mediumseagreen"
        else:
            str_msg = str_msg + "\n\n   SYSTEM UNACCEPTABLE\n"
            str_color = "red"

        ax3.annotate(
            str_msg,
            xy=(0.4, 0.1),
            bbox=dict(boxstyle="round", fc=str_color, color="black"),
            fontsize=12,
        )

        str_note = "Concordance analysis take care the operator\n"
        str_note = str_note + " answer the same to the same pieze,\n"
        str_note = str_note + " being equal to the reference"
        ax3.annotate(str_note, xy=(0.65, 0.6), fontsize=10)

        # ==============================================
        #          Confusion Matrix
        # ==============================================
        df_0 = self.__df_0
        dict_Matrix = dict()
        for operator in df_0["Op"].unique().tolist():
            dict_Matrix[operator] = {
                "0_0": len(
                    df_0[
                        (df_0["Op"] == operator)
                        & (df_0["Acc"] == True)  # noqa: E712
                        & (df_0["Ref"] == df_0["Ref"].unique().tolist()[0])
                    ]
                )
                / self.__p
                / self.__r
                * 100,
                "0_1": len(
                    df_0[
                        (df_0["Op"] == operator)
                        & (df_0["Acc"] == True)  # noqa: E712
                        & (df_0["Ref"] == df_0["Ref"].unique().tolist()[1])
                    ]
                )
                / self.__p
                / self.__r
                * 100,
                "1_0": len(
                    df_0[
                        (df_0["Op"] == operator)
                        & (df_0["Acc"] == False)  # noqa: E712
                        & (df_0["Ref"] == df_0["Ref"].unique().tolist()[0])
                    ]
                )
                / self.__p
                / self.__r
                * 100,
                "1_1": len(
                    df_0[
                        (df_0["Op"] == operator)
                        & (df_0["Acc"] == False)  # noqa: E712
                        & (df_0["Ref"] == df_0["Ref"].unique().tolist()[1])
                    ]
                )
                / self.__p
                / self.__r
                * 100,
            }

        df_matrix = pd.DataFrame(dict_Matrix)
        df_1 = self.__df_1

        ax4 = Fig_R.add_subplot(gs[2:4, 0:2])
        ax4.set_title("Confusion Matrix", fontweight="bold")
        ax4.set_facecolor("white")
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_ylim(0, 1)
        ax4.set_xlim(0, 1)

        df_iMatrix = df_matrix.T
        ax41 = Fig_R.add_subplot(gs[2, 0])
        ax41.set_ylabel("FALSE")
        ax41.set_facecolor("white")
        ax41.barh(df_iMatrix.index, df_iMatrix["0_1"])
        ax41.set_xlim(0, 100)
        ax41.set_xticks([])
        ax41.yaxis.label.set_color("red")

        ax42 = Fig_R.add_subplot(gs[3, 0])
        ax42.set_xlabel("TRUE")
        ax42.set_ylabel("TRUE")
        ax42.set_facecolor("white")
        ax42.barh(df_iMatrix.index, df_iMatrix["0_0"])
        ax42.set_xlim(0, 100)
        ax42.set_xticks([])
        ax42.xaxis.label.set_color("red")
        ax42.yaxis.label.set_color("red")

        ax43 = Fig_R.add_subplot(gs[2, 1])
        ax43.set_facecolor("white")
        ax43.barh(df_iMatrix.index, df_iMatrix["1_1"])
        ax43.set_xlim(0, 100)
        ax43.set_xticks([])

        ax44 = Fig_R.add_subplot(gs[3, 1])
        ax44.set_xlabel("FALSE")
        ax44.set_facecolor("white")
        ax44.barh(df_iMatrix.index, df_iMatrix["1_0"])
        ax44.set_xlim(0, 100)
        ax44.set_xticks([])
        ax44.xaxis.label.set_color("red")
        # ==============================================
        #          Accuracy Analysis
        # ==============================================

        df_1.sort_index(inplace=True)
        ax5 = Fig_R.add_subplot(gs[2:4, 2:4])
        ax5.set_title("Accuracy based on % matches", fontweight="bold")
        ax5.barh(df_1.index, df_1["Acc"])
        ax5.set_xlim(0, 100)
        ax5.axvline(
            df_0["Acc"].value_counts()[True] / len(df_0.index) * 100,
            color="red",
            linestyle="--",
            label=(
                f"{df_0['Acc'].value_counts()[True]/len(df_0.index)*100:.1f}"
                f"% | SYSTEM"
            ),
        )
        ax5.legend()

        # ==============================================
        #          Accuracy msg
        # ==============================================
        ax6 = Fig_R.add_subplot(gs[4, :])
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_facecolor("white")
        ax6.set_ylim(0, 1)
        ax6.set_xlim(0, 1)

        dbl_value = df_0["Acc"].value_counts()[True] / len(df_0.index) * 100
        str_msg = (
            f"\n  Accuracy analysis:\n"
            f"      Reproducibility  | {dbl_value:.1f}%  "
        )

        if df_0["Acc"].value_counts()[True] / len(df_0.index) * 100 > 80:
            str_msg = str_msg + "\n\n   SYSTEM ACCEPTABLE\n"
            str_color = "mediumseagreen"
        else:
            str_msg = str_msg + "\n\n   SYSTEM UNACCEPTABLE\n"
            str_color = "red"

        ax6.annotate(
            str_msg,
            xy=(0.4, 0.1),
            bbox=dict(boxstyle="round", fc=str_color, color="black"),
            fontsize=12,
        )

        str_note = "Accuracy analysis sum all answer equal to the reference\n"
        str_note = str_note + " It doe not take care of the repeteability"

        ax6.annotate(str_note, xy=(0.65, 0.6), fontsize=10)
