# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-19

测试AttackedText以及ChineseAttackedText类
"""


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

from text.EvalBox.Attack.attack_eval import AttackEval
from text.EvalBox.Attack.attack_args import AttackArgs
from text.EvalBox.Attack.text_fooler import TextFooler
from text.Datasets import SST


def run():
    """ """
    ds = SST()
    txt_fooler = TextFooler(language="en", verbose=2)
    aa = AttackArgs(log_to_txt=True)
    ae = AttackEval(txt_fooler, ds, aa)
    ae.attack_dataset()


if __name__ == "__main__":
    run()
