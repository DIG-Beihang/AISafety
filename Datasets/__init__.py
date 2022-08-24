# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-16
@LastEditTime: 2021-09-23
"""

from .base import NLPDataset
from .amazon_reviews_zh import AmazonReviewsZH
from .sst import SST
from .imdb_reviews_tiny import IMDBReviewsTiny
from .dianping_tiny import DianPingTiny
from .jd_binary_tiny import JDBinaryTiny
from .jd_full_tiny import JDFullTiny
from .ifeng import Ifeng
from .chinanews import Chinanews


__all__ = [
    "NLPDataset",
    "AmazonReviewsZH",
    "SST",
    "IMDBReviewsTiny",
    "DianPingTiny",
    "JDBinaryTiny",
    "JDFullTiny",
    "Ifeng",
    "Chinanews",
]
