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

import numpy as np
import pandas as pd


class Stat_Tables:
    """Statistic tables paremeter"""

    def __init__(self):
        """Initializate tables object loading
        tables parameter into a dictionary

        Methods:
        ---------
        get_A : Float
            Return the A constant

        get_A2 : Float
            Return the A2 constant

        get_A3 : Float
            Return the A3 constant

        get_c4 : Float
            Return the c4 constant

        get_B3 : Float
            Return the B3 constant

        get_B4 : Float
            Return the B4 constant

        get_B5 : Float
            Return the B5 constant

        get_B6 : Float
            Return the B6 constant

        get_B4 : Float
            Return the B4 constant

        get_d2 : Float
            Return the d2 constant

        get_d3 : Float
            Return the d3 constant

        get_D1 : Float
            Return the D1 constant

        get_D2 : Float
            Return the D2 constant

        get_D3 : Float
            Return the D3 constant

        get_D4 : Float
            Return the D4 constant
        """
        self.__dict_info = {
            "A": {
                2: 2.121,
                3: 1.732,
                4: 1.5,
                5: 1.342,
                6: 1.225,
                7: 1.134,
                8: 1.061,
                9: 1,
                10: 0.949,
                11: 0.905,
                12: 0.866,
                13: 0.832,
                14: 0.802,
                15: 0.775,
                16: 0.75,
                17: 0.728,
                18: 0.707,
                19: 0.688,
                20: 0.671,
                21: 0.655,
                22: 0.64,
                23: 0.626,
                24: 0.612,
                25: 0.6,
            },
            "A2": {
                2: 1.88,
                3: 1.023,
                4: 0.729,
                5: 0.577,
                6: 0.483,
                7: 0.419,
                8: 0.373,
                9: 0.337,
                10: 0.308,
                11: 0.285,
                12: 0.266,
                13: 0.249,
                14: 0.235,
                15: 0.223,
                16: 0.212,
                17: 0.203,
                18: 0.194,
                19: 0.187,
                20: 0.18,
                21: 0.173,
                22: 0.167,
                23: 0.162,
                24: 0.157,
                25: 0.153,
            },
            "A3": {
                2: 2.659,
                3: 1.954,
                4: 1.628,
                5: 1.427,
                6: 1.287,
                7: 1.182,
                8: 1.099,
                9: 1.032,
                10: 0.975,
                11: 0.927,
                12: 0.886,
                13: 0.85,
                14: 0.817,
                15: 0.789,
                16: 0.763,
                17: 0.739,
                18: 0.718,
                19: 0.698,
                20: 0.68,
                21: 0.663,
                22: 0.647,
                23: 0.633,
                24: 0.619,
                25: 0.606,
            },
            "c4": {
                2: 0.7979,
                3: 0.8862,
                4: 0.9213,
                5: 0.94,
                6: 0.9515,
                7: 0.9594,
                8: 0.965,
                9: 0.9693,
                10: 0.9727,
                11: 0.9754,
                12: 0.9776,
                13: 0.9794,
                14: 0.981,
                15: 0.9823,
                16: 0.9835,
                17: 0.9845,
                18: 0.9854,
                19: 0.9862,
                20: 0.9869,
                21: 0.9876,
                22: 0.9882,
                23: 0.9887,
                24: 0.9892,
                25: 0.9896,
            },
            "B3": {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0.03,
                7: 0.118,
                8: 0.185,
                9: 0.239,
                10: 0.284,
                11: 0.321,
                12: 0.354,
                13: 0.382,
                14: 0.406,
                15: 0.428,
                16: 0.448,
                17: 0.466,
                18: 0.482,
                19: 0.497,
                20: 0.51,
                21: 0.523,
                22: 0.534,
                23: 0.545,
                24: 0.555,
                25: 0.565,
            },
            "B4": {
                2: 3.267,
                3: 2.568,
                4: 2.266,
                5: 2.089,
                6: 1.97,
                7: 1.882,
                8: 1.815,
                9: 1.761,
                10: 1.716,
                11: 1.679,
                12: 1.646,
                13: 1.618,
                14: 1.594,
                15: 1.572,
                16: 1.552,
                17: 1.534,
                18: 1.518,
                19: 1.503,
                20: 1.49,
                21: 1.477,
                22: 1.466,
                23: 1.455,
                24: 1.445,
                25: 1.435,
            },
            "B5": {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0.029,
                7: 0.113,
                8: 0.179,
                9: 0.232,
                10: 0.276,
                11: 0.313,
                12: 0.346,
                13: 0.374,
                14: 0.399,
                15: 0.421,
                16: 0.44,
                17: 0.458,
                18: 0.475,
                19: 0.49,
                20: 0.504,
                21: 0.516,
                22: 0.528,
                23: 0.539,
                24: 0.549,
                25: 0.559,
            },
            "B6": {
                2: 2.606,
                3: 2.276,
                4: 2.088,
                5: 1.964,
                6: 1.874,
                7: 1.806,
                8: 1.751,
                9: 1.707,
                10: 1.669,
                11: 1.637,
                12: 1.61,
                13: 1.585,
                14: 1.563,
                15: 1.544,
                16: 1.526,
                17: 1.511,
                18: 1.496,
                19: 1.483,
                20: 1.47,
                21: 1.459,
                22: 1.448,
                23: 1.438,
                24: 1.429,
                25: 1.42,
            },
            "d2": {
                2: 1.128,
                3: 1.693,
                4: 2.059,
                5: 2.326,
                6: 2.534,
                7: 2.704,
                8: 2.847,
                9: 2.97,
                10: 3.078,
                11: 3.173,
                12: 3.258,
                13: 3.336,
                14: 3.407,
                15: 3.472,
                16: 3.532,
                17: 3.588,
                18: 3.64,
                19: 3.689,
                20: 3.735,
                21: 3.778,
                22: 3.819,
                23: 3.858,
                24: 3.895,
                25: 3.931,
            },
            "d3": {
                2: 0.852,
                3: 0.888,
                4: 0.879,
                5: 0.864,
                6: 0.848,
                7: 0.833,
                8: 0.819,
                9: 0.807,
                10: 0.797,
                11: 0.787,
                12: 0.778,
                13: 0.77,
                14: 0.763,
                15: 0.756,
                16: 0.75,
                17: 0.744,
                18: 0.738,
                19: 0.733,
                20: 0.728,
                21: 0.724,
                22: 0.719,
                23: 0.715,
                24: 0.712,
                25: 0.708,
            },
            "D1": {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0.206,
                8: 0.389,
                9: 0.548,
                10: 0.688,
                11: 0.813,
                12: 0.924,
                13: 1.026,
                14: 1.119,
                15: 1.204,
                16: 1.283,
                17: 1.357,
                18: 1.425,
                19: 1.49,
                20: 1.55,
                21: 1.607,
                22: 1.661,
                23: 1.712,
                24: 1.761,
                25: 1.807,
            },
            "D2": {
                2: 3.686,
                3: 4.357,
                4: 4.697,
                5: 4.918,
                6: 5.078,
                7: 5.203,
                8: 5.306,
                9: 5.392,
                10: 5.467,
                11: 5.533,
                12: 5.593,
                13: 5.646,
                14: 5.695,
                15: 5.739,
                16: 5.781,
                17: 5.819,
                18: 5.855,
                19: 5.888,
                20: 5.92,
                21: 5.95,
                22: 5.978,
                23: 6.004,
                24: 6.03,
                25: 6.055,
            },
            "D3": {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0.076,
                8: 0.137,
                9: 0.184,
                10: 0.223,
                11: 0.256,
                12: 0.284,
                13: 0.307,
                14: 0.328,
                15: 0.347,
                16: 0.363,
                17: 0.378,
                18: 0.392,
                19: 0.404,
                20: 0.415,
                21: 0.425,
                22: 0.435,
                23: 0.444,
                24: 0.452,
                25: 0.46,
            },
            "D4": {
                2: 3.266,
                3: 2.574,
                4: 2.281,
                5: 2.114,
                6: 2.003,
                7: 1.924,
                8: 1.863,
                9: 1.816,
                10: 1.777,
                11: 1.744,
                12: 1.716,
                13: 1.693,
                14: 1.672,
                15: 1.653,
                16: 1.637,
                17: 1.622,
                18: 1.608,
                19: 1.596,
                20: 1.585,
                21: 1.575,
                22: 1.565,
                23: 1.556,
                24: 1.548,
                25: 1.54,
            },
        }

    def get_A(self, n):
        """Return the A constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : A constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["A"][n]
        else:
            return round(3 / np.sqrt(n), 4)

    def get_A2(self, n, bol_Pass=False):
        """Return the A2 constant

        Args:
        ------
        n : int
            Number of items

        bol_Pass: Boolean
            False to raise error for n>25, True for A2_25 when n>25

        Returns
        -------
        Float : A2 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2

        A_03
            Number of items is higher than 25. Set bol_Pass=True to avoid
            this error and alue for n=25 wil be returned
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items number must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(
                f"Error A_02: Samples must be >=2. Specified: {n}"
            )
        elif n > 25 and bol_Pass is False:
            raise ValueError(
                f"Error A_03: items number is higher than 25. Specified: {n}."
                + " Set bol_Pass=True to avoid this error and alue for n=25"
                + " will be returned"
            )
        elif n > 25 and bol_Pass is True:
            return self.__dict_info["A2"][25]
        else:
            return self.__dict_info["A2"][n]

    def get_c4(self, n):
        """Return the c4 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : c4 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["c4"][n]
        else:
            return round(4 * (n - 1) / (4 * n - 3), 4)

    def get_A3(self, n):
        """Return the A3 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : A3 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["A3"][n]
        else:
            return round(3 / (self.get_c4(n) * np.sqrt(n)), 4)

    def get_B3(self, n):
        """Return the B3 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : B3 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["B3"][n]
        else:
            return round(1 - 3 / (self.get_c4(n) * np.sqrt(2 * (n - 1))), 4)

    def get_B4(self, n):
        """Return the B4 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : B4 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["B4"][n]
        else:
            return round(1 + 3 / (self.get_c4(n) * np.sqrt(2 * (n - 1))), 4)

    def get_B5(self, n):
        """Return the B5 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : B5 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["B5"][n]
        else:
            return round(self.get_c4(n) - 3 / np.sqrt(2 * (n - 1)), 4)

    def get_B6(self, n):
        """Return the B6 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : B6 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["B6"][n]
        else:
            return round(self.get_c4(n) + 3 / np.sqrt(2 * (n - 1)), 4)

    def get_d2(self, n):
        """Return the d2 constant

        Args:
        ------
        n : int
            Number of items

        Returns
        -------
        Float : d2 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(f"Error A_02: Items must be >=2. Specified: {n}")
        elif n >= 2 and n <= 25:
            return self.__dict_info["d2"][n]
        else:
            mu, sigma = 0, 1
            x = pd.DataFrame()
            for i in range(1, n + 1):
                x["x" + str(i)] = np.random.normal(mu, sigma, 10000000)
            x["range"] = abs(x.min(axis=1) - x.max(axis=1))
            return round(x["range"].mean(axis=0).round(3), 4)

    def get_d3(self, n, bol_Pass=False):
        """Return the d3 constant

        Args:
        ------
        n : int
            Number of items

        bol_Pass: Boolean
            False to raise error for n>25, True for d3_25 when n>25

        Returns
        -------
        Float : d3 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2

        A_03
            Number of items is higher than 25. Set bol_Pass=True to avoid
            this error and alue for n=25 wil be returned
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items number must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(
                f"Error A_02: Samples must be >=2. Specified: {n}"
            )
        elif n > 25 and bol_Pass is False:
            raise ValueError(
                f"Error A_03: Items number is higher than 25. Specified: {n}."
                + " Set bol_Pass=True to avoid this error and alue for n=25"
                + " will be returned"
            )
        elif n > 25 and bol_Pass is True:
            return self.__dict_info["d3"][25]
        else:
            return self.__dict_info["d3"][n]

    def get_D1(self, n, bol_Pass=False):
        """Return the D1 constant

        Args:
        ------
        n : int
            Number of items

        bol_Pass: Boolean
            False to raise error for n>25, True for D1_25 when n>25

        Returns
        -------
        Float : D1 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2

        A_03
            Number of items is higher than 25. Set bol_Pass=True to avoid
            this error and alue for n=25 wil be returned
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items number must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(
                f"Error A_02: Samples must be >=2. Specified: {n}"
            )
        elif n > 25 and bol_Pass is False:
            raise ValueError(
                f"Error A_03: Items number is higher than 25. Specified: {n}."
                + " Set bol_Pass=True to avoid this error and alue for n=25"
                + " will be returned"
            )
        elif n > 25 and bol_Pass is True:
            return self.__dict_info["D1"][25]
        else:
            return self.__dict_info["D1"][n]

    def get_D2(self, n, bol_Pass=False):
        """Return the D2 constant

        Args:
        ------
        n : int
            Number of items

        bol_Pass: Boolean
            False to raise error for n>25, True for D2_25 when n>25

        Returns
        -------
        Float : D2 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2

        A_03
            Number of items is higher than 25. Set bol_Pass=True to avoid
            this error and alue for n=25 wil be returned
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items number must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(
                f"Error A_02: Samples must be >=2. Specified: {n}"
            )
        elif n > 25 and bol_Pass is False:
            raise ValueError(
                f"Error A_03: Items number is higher than 25. Specified: {n}."
                + " Set bol_Pass=True to avoid this error and alue for n=25"
                + " will be returned"
            )
        elif n > 25 and bol_Pass is True:
            return self.__dict_info["D2"][25]
        else:
            return self.__dict_info["D2"][n]

    def get_D3(self, n, bol_Pass=False):
        """Return the D3 constant

        Args:
        ------
        n : int
            Number of items

        bol_Pass: Boolean
            False to raise error for n>25, True for D3_25 when n>25

        Returns
        -------
        Float : D3 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2

        A_03
            Number of items is higher than 25. Set bol_Pass=True to avoid
            this error and alue for n=25 wil be returned
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items number must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(
                f"Error A_02: Samples must be >=2. Specified: {n}"
            )
        elif n > 25 and bol_Pass is False:
            raise ValueError(
                f"Error A_03: Items number is higher than 25. Specified: {n}."
                + " Set bol_Pass=True to avoid this error and alue for n=25"
                + " will be returned"
            )
        elif n > 25 and bol_Pass is True:
            return self.__dict_info["D3"][25]
        else:
            return self.__dict_info["D3"][n]

    def get_D4(self, n, bol_Pass=False):
        """Return the D4 constant

        Args:
        ------
        n : int
            Number of items

        bol_Pass: Boolean
            False to raise error for n>25, True for D4_25 when n>25

        Returns
        -------
        Float : D4 constant for the given number of items

        Raises:
        --------
        TypeError

        A_01
            n must be an integer

        A_02
            Number of items must be >=2

        A_03
            Number of items is higher than 25. Set bol_Pass=True to avoid
            this error and alue for n=25 wil be returned
        """
        if type(n) is not int:
            raise ValueError(
                f"Error A_01: Items number must be an integer. Specified: {n}"
            )
        if n < 2:
            raise ValueError(
                f"Error A_02: Samples must be >=2. Specified: {n}"
            )
        elif n > 25 and bol_Pass is False:
            raise ValueError(
                f"Error A_03: Items number is higher than 25. Specified: {n}."
                + " Set bol_Pass=True to avoid this error and alue for n=25"
                + " will be returned"
            )
        elif n > 25 and bol_Pass is True:
            return self.__dict_info["D4"][25]
        else:
            return self.__dict_info["D4"][n]
