# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-22
@LastEditTime: 2022-04-01

预置模型，计划有WordCNN, LSTM等
"""

from .bert_amazon_zh import VictimBERTAmazonZH
from .lstm_for_classification import VictimLSTMForClassification
from .roberta_chinanews import VictimRoBERTaChinaNews
from .roberta_dianping import VictimRoBERTaDianPing
from .roberta_ifeng import VictimRoBERTaIFeng
from .roberta_sst import VictimRoBERTaSST
from .word_cnn_for_classification import VictimWordCNNForClassification
