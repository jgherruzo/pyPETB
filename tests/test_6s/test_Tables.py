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

import logging
import random

from pypetb import tables

logging.basicConfig(
    filename="test_stat_table.log", encoding="utf-8", level=logging.DEBUG
)


def test_A_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A_01 with {n}")
    try:
        tbl.get_A(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A_02 with {n}")
    try:
        tbl.get_A(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A_03 with {n}")
    test = tbl.get_A(n)
    assert test == float(test)


def test_A2_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A2_01 with {n}")
    try:
        tbl.get_A2(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A2_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A2_02 with {n}")
    try:
        tbl.get_A2(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A2_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A2_03 with {n}")
    if n > 25:
        test = tbl.get_A2(n, bol_Pass=True)
    else:
        test = tbl.get_A2(n)
    assert test == float(test)


def test_A2_04():
    """Test correct item number"""
    tbl = tables.Stat_Tables()
    logging.debug("Test test_A2_04 with 26")
    try:
        tbl.get_A2(26)
    except ValueError as error:
        if "A_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A3_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A3_01 with {n}")
    try:
        tbl.get_A3(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A3_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A3_02 with {n}")
    try:
        tbl.get_A3(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_A3_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_A3_03 with {n}")
    test = tbl.get_A3(n)
    assert test == float(test)


def test_c4_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_c4_01 with {n}")
    try:
        tbl.get_c4(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_c4_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_c4_02 with {n}")
    try:
        tbl.get_c4(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_c4_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_c4_03 with {n}")
    test = tbl.get_c4(n)
    assert test == float(test)


def test_B3_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B3_01 with {n}")
    try:
        tbl.get_B3(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B3_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B3_02 with {n}")
    try:
        tbl.get_B3(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B3_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B3_03 with {n}")
    test = tbl.get_B3(n)
    assert test == float(test)


def test_B4_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B4_01 with {n}")
    try:
        tbl.get_B4(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B4_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B4_02 with {n}")
    try:
        tbl.get_B4(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B4_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B4_03 with {n}")
    test = tbl.get_B4(n)
    assert test == float(test)


def test_B5_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B5_01 with {n}")
    try:
        tbl.get_B5(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B5_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B5_02 with {n}")
    try:
        tbl.get_B5(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B5_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B5_03 with {n}")
    test = tbl.get_B5(n)
    assert test == float(test)


def test_B6_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B6_01 with {n}")
    try:
        tbl.get_B6(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B6_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B6_02 with {n}")
    try:
        tbl.get_B6(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_B6_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_B6_03 with {n}")
    test = tbl.get_B6(n)
    assert test == float(test)


def test_d2_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_d2_01 with {n}")
    try:
        tbl.get_d2(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_d2_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_d2_02 with {n}")
    try:
        tbl.get_d2(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_d2_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_d2_03 with {n}")
    test = tbl.get_d2(n)
    assert test == float(test)


def test_d3_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_d3_01 with {n}")
    try:
        tbl.get_d3(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_d3_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_d3_02 with {n}")
    try:
        tbl.get_d3(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_d3_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_d3_03 with {n}")
    if n > 25:
        test = tbl.get_d3(n, bol_Pass=True)
    else:
        test = tbl.get_d3(n)
    assert test == float(test)


def test_d3_04():
    """Test correct item number"""
    tbl = tables.Stat_Tables()
    logging.debug("Test test_d3_04 with 26")
    try:
        tbl.get_d3(26)
    except ValueError as error:
        if "A_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D1_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D1_01 with {n}")
    try:
        tbl.get_D1(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D1_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D1_02 with {n}")
    try:
        tbl.get_D1(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D1_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D1_03 with {n}")
    if n > 25:
        test = tbl.get_D1(n, bol_Pass=True)
    else:
        test = tbl.get_D1(n)
    assert test == float(test)


def test_D1_04():
    """Test correct item number"""
    tbl = tables.Stat_Tables()
    logging.debug("Test test_D1_04 with 26")
    try:
        tbl.get_D1(26)
    except ValueError as error:
        if "A_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D2_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D2_01 with {n}")
    try:
        tbl.get_D2(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D2_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D2_02 with {n}")
    try:
        tbl.get_D2(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D2_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D2_03 with {n}")
    if n > 25:
        test = tbl.get_D2(n, bol_Pass=True)
    else:
        test = tbl.get_D2(n)
    assert test == float(test)


def test_D2_04():
    """Test correct item number"""
    tbl = tables.Stat_Tables()
    logging.debug("Test test_D2_04 with 26")
    try:
        tbl.get_D2(26)
    except ValueError as error:
        if "A_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D3_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D3_01 with {n}")
    try:
        tbl.get_D3(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D3_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D3_02 with {n}")
    try:
        tbl.get_D3(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D3_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D3_03 with {n}")
    if n > 25:
        test = tbl.get_D3(n, bol_Pass=True)
    else:
        test = tbl.get_D3(n)
    assert test == float(test)


def test_D3_04():
    """Test correct item number"""
    tbl = tables.Stat_Tables()
    logging.debug("Test test_D3_04 with 26")
    try:
        tbl.get_D3(26)
    except ValueError as error:
        if "A_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D4_01():
    """Test correct data type"""
    n = None
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D4_01 with {n}")
    try:
        tbl.get_D4(n)
    except ValueError as error:
        if "A_01" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D4_02():
    """Test correct item number"""
    n = random.randint(-100, 1)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D4_02 with {n}")
    try:
        tbl.get_D4(n)
    except ValueError as error:
        if "A_02" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True


def test_D4_03():
    """Test correct item number"""
    n = random.randint(2, 100)
    tbl = tables.Stat_Tables()
    logging.debug(f"Test test_D4_03 with {n}")
    if n > 25:
        test = tbl.get_D4(n, bol_Pass=True)
    else:
        test = tbl.get_D4(n)
    assert test == float(test)


def test_D4_04():
    """Test correct item number"""
    tbl = tables.Stat_Tables()
    logging.debug("Test test_D4_04 with 26")
    try:
        tbl.get_D4(26)
    except ValueError as error:
        if "A_03" in str(error):
            bol_temp = True
        else:
            bol_temp = False

    assert bol_temp is True
