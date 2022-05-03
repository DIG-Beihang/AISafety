# import os
# import pkgutil
#
# pkgpath = os.path.dirname(__file__)
# pkgname = os.path.basename(pkgpath)
# for _, file, _ in pkgutil.iter_modules([pkgpath]):
#     print(pkgname,file)
#     #from Models.TestModel.+'file'
#     __import__('Models.'+pkgname+'.'+file)

from .bert_amazon_zh import VictimBERTAmazonZH
from .lstm_for_classification import VictimLSTMForClassification
from .roberta_chinanews import VictimRoBERTaChinaNews
from .roberta_dianping import VictimRoBERTaDianPing
from .roberta_ifeng import VictimRoBERTaIFeng
from .roberta_sst import VictimRoBERTaSST
from .word_cnn_for_classification import VictimWordCNNForClassification