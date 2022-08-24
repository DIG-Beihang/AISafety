"""
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text.Datasets.amazon_reviews_zh import AmazonReviewsZH
from text.Datasets.sst import SST

from text.EvalBox.Attack.text_fooler import TextFooler
from text.EvalBox.Attack.pwws import PWWS  # noqa: F401

from text.EvalBox.Attack.attack_eval import AttackEval
from text.EvalBox.Attack.attack_args import AttackArgs


ds_amazon_reviews_zh = AmazonReviewsZH()
ds_sst = SST()

text_fooler_en = TextFooler(language="en", verbose=2)
aa = AttackArgs(
    num_examples=7,
    log_to_txt=True,
    log_adv_gen=False,
)
ae = AttackEval(text_fooler_en, ds_sst, aa)
ae.attack_dataset()


# too slow
# text_fooler_zh = TextFooler(language="zh", verbose=2)
# example, label = ds_amazon_reviews_zh[1]
# text_fooler_zh.attack(example, label)
