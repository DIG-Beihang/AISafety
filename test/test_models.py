# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-29
@LastEditTime: 2021-10-04

测试内置Victim模型
"""

import random
import time

if __name__ == "__main__" and __package__ is None:
    level = 2
    # https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
    import sys
    import importlib
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass
    __package__ = ".".join(parent.parts[len(top.parts) :])
    importlib.import_module(__package__)  # won't be needed after that

from text.Models.TestModel.bert_amazon_zh import VictimBERTAmazonZH
from text.Models.TestModel.roberta_sst import VictimRoBERTaSST
from text.Models.TestModel.roberta_dianping import VictimRoBERTaDianPing
from text.Models.TestModel.word_cnn_for_classification import (
    WordCNNForClassification,
)
from text.Models.TestModel.lstm_for_classification import LSTMForClassification

from text.EvalBox.Attack.text_fooler import TextFooler  # noqa: F401
from text.EvalBox.Attack.pwws import PWWS

from text.Datasets.amazon_reviews_zh import AmazonReviewsZH
from text.Datasets.sst import SST
from text.Datasets.imdb_reviews_tiny import IMDBReviewsTiny
from text.Datasets.dianping_tiny import DianPingTiny
from text.Datasets.jd_full_tiny import JDFullTiny


def run_test():
    """ """
    method = PWWS  # TextFooler
    # datasets
    print("\n" + " running models test ".center(100, "#") + "\n")
    print("loading datasets...\n")
    start = time.time()
    arzh = AmazonReviewsZH()
    sst = SST()
    imdb = IMDBReviewsTiny()
    dp = DianPingTiny()
    jd_full = JDFullTiny()
    print(f"datasets loaded in {time.time()-start:.2f} seconds\n")

    # victim model
    # 1. VictimBERTAmazonZH
    print(
        "\n"
        + f" testing VictimBERTAmazonZH using {method.__name__} ".center(80, "-")
        + "\n"
    )
    start = time.time()
    model = VictimBERTAmazonZH()
    attacker = method(model=model, language="zh", verbose=2)
    attacker.cuda_if_possible_()
    samples = random.sample(list(range(len(arzh))), 5)

    print("start attacking...")
    for idx in samples:
        print("*" * 40)
        sample, label = arzh[idx]
        att_res = attacker.attack(sample, label)
        print("\nattack result:")
        print(att_res)
        print("*" * 40 + "\n")

    # VictimBERTAmazonZH 模型是在amazon商品评论数据集上训练的
    # 看看在京东商品评论数据集上的效果
    samples = random.sample(list(range(len(jd_full))), 5)
    for idx in samples:
        print("*" * 40)
        sample, label = jd_full[idx]
        att_res = attacker.attack(sample, label)
        print("\nattack result:")
        print(att_res)
        print("*" * 40 + "\n")

    print(f"testing of VictimBERTAmazonZH used {time.time()-start:.2f} seconds")

    # 2. VictimRoBERTaSST
    print(
        "\n"
        + f" testing VictimRoBERTaSST using {method.__name__} ".center(80, "-")
        + "\n"
    )
    start = time.time()
    model = VictimRoBERTaSST()
    attacker = method(model=model, language="en", verbose=2)
    attacker.cuda_if_possible_()
    samples = random.sample(list(range(len(sst))), 5)

    print("start attacking...")
    for idx in samples:
        print("*" * 40)
        sample, label = sst[idx]
        att_res = attacker.attack(sample, label)
        print("\nattack result:")
        print(att_res)
        print("*" * 40 + "\n")

    print(f"testing of VictimRoBERTaSST used {time.time()-start:.2f} seconds")

    # 3. VictimRoBERTaDianPing
    print(
        "\n"
        + f" testing VictimRoBERTaDianPing using {method.__name__} ".center(80, "-")
        + "\n"
    )
    start = time.time()
    model = VictimRoBERTaDianPing()
    attacker = method(model=model, language="zh", verbose=2)
    attacker.cuda_if_possible_()
    samples = random.sample(list(range(len(dp))), 5)

    print("start attacking...")
    for idx in samples:
        print("*" * 40)
        sample, label = dp[idx]
        att_res = attacker.attack(sample, label)
        print("\nattack result:")
        print(att_res)
        print("*" * 40 + "\n")

    print(f"testing of VictimRoBERTaDianPing used {time.time()-start:.2f} seconds")

    # 4. WordCNNForClassification
    print(
        "\n"
        + f" testing WordCNNForClassification using {method.__name__} ".center(80, "-")
        + "\n"
    )
    start = time.time()
    model = WordCNNForClassification("cnn-imdb")
    attacker = method(model=model, language="en", verbose=2)
    attacker.cuda_if_possible_()
    samples = random.sample(list(range(len(imdb))), 5)

    print("start attacking...")
    for idx in samples:
        print("*" * 40)
        sample, label = imdb[idx]
        att_res = attacker.attack(sample, label)
        print("\nattack result:")
        print(att_res)
        print("*" * 40 + "\n")

    print(f"testing of WordCNNForClassification used {time.time()-start:.2f} seconds")

    # 5. LSTMForClassification
    print(
        "\n"
        + f" testing LSTMForClassification using {method.__name__} ".center(80, "-")
        + "\n"
    )
    start = time.time()
    model = LSTMForClassification("lstm-sst2")
    attacker = method(model=model, language="en", verbose=2)
    attacker.cuda_if_possible_()
    samples = random.sample(list(range(len(sst))), 5)

    print("start attacking...")
    for idx in samples:
        print("*" * 40)
        sample, label = sst[idx]
        att_res = attacker.attack(sample, label)
        print("\nattack result:")
        print(att_res)
        print("*" * 40 + "\n")

    print(f"testing of LSTMForClassification used {time.time()-start:.2f} seconds")

    print("\n" + " models test finishes ".center(100, "#") + "\n")


if __name__ == "__main__":
    run_test()
