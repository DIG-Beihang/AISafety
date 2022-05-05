# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-09-19

测试各种constraint类
"""


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
