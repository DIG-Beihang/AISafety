# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-14
@LastEditTime: 2021-09-19
"""

import random


if __name__ == "__main__" and __package__ is None:
    level = 3
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

from text.Datasets.amazon_reviews_zh import AmazonReviewsZH
from text.Datasets.sst import SST
from text.EvalBox.Attack.faster_genetic import FasterGenetic


def run(test_num: int = 3):
    """ """
    ds_arzh = AmazonReviewsZH()
    ds_sst = SST()

    attacker_en = FasterGenetic(language="en", verbose=2)
    attacker_en.cuda_if_possible_()

    sample_inds = random.sample(range(len(ds_sst)), test_num)
    print(f"sample indices of SST = {sample_inds}")
    for idx in sample_inds:
        sample, label = ds_sst[idx]
        res = attacker_en.attack(sample, label)
        print(f"{idx+1}-th attack result is \n{res}")
    del attacker_en

    attacker_zh = FasterGenetic(language="zh", verbose=2)
    attacker_zh.cuda_if_possible_()

    sample_inds = random.sample(range(len(ds_arzh)), test_num)
    print(f"sample indices of AmazonReviewsZH = {sample_inds}")
    for idx in sample_inds:
        sample, label = ds_arzh[idx]
        res = attacker_zh.attack(sample, label)
        print(f"{idx+1}-th attack result is \n{res}")
    del attacker_zh


if __name__ == "__main__":
    run()
