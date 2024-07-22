# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/3/17 23:30
# @Function:
import enum


class Environment(enum.Enum):
    n20db = -20
    p0db = 0
    p20db = 20
    # Lest012‘s value never used since their snr are defined differently， select any other numbers different from above
    Lest1 = 1
    Lest2 = 2
    Lest3 = 3


