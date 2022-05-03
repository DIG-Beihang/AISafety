# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-07-30
@LastEditTime: 2022-04-17

"""

import os
import subprocess
import collections
import time
import platform
import shlex
import random
import signal
from contextlib import contextmanager
from logging import Logger
from numbers import Real, Number
from itertools import groupby, product
from operator import itemgetter
from typing import MutableMapping, Union, Any, Optional, List, Tuple, NoReturn, Sequence

import numpy as np
import pandas as pd
import torch


__all__ = [
    "DEFAULTS",
    "set_seed",
    "default_device",
    "module_dir",
    "nlp_cache_dir",
    "nlp_log_dir",
    "execute_cmd",
    "dict_to_str",
    "dicts_equal",
    "hashable",
    "consecutive_groups",
    "dataframe_selection",
    "str2bool",
    "timeout",
]


module_dir = os.path.abspath(__file__)
for _ in range(2):
    module_dir = os.path.dirname(module_dir)
module_name = os.path.basename(module_dir)
# project_dir = os.path.dirname(module_dir)

nlp_cache_dir = os.path.join(module_dir, "data")
nlp_log_dir = os.path.join(module_dir, "log")

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFG(dict):
    """

    this class is created in order to renew the `update` method,
    to fit the hierarchical structure of configurations

    Examples
    --------
    >>> c = CFG(hehe={"a":1,"b":2})
    >>> c.update(hehe={"a":-1})
    >>> c
    {'hehe': {'a': -1, 'b': 2}}
    >>> c.__update__(hehe={"a":-10})
    >>> c
    {'hehe': {'a': -10}}

    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> NoReturn:
        """ """
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            d = args[0]
            assert isinstance(d, MutableMapping)
        else:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            try:
                setattr(self, k, v)
            except Exception:
                dict.__setitem__(self, k, v)
        # Class attributes
        exclude_fields = ["update", "pop"]
        for k in self.__class__.__dict__:
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in exclude_fields
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(
        self, new_cfg: Optional[MutableMapping] = None, **kwargs: Any
    ) -> NoReturn:
        """

        the new hierarchical update method

        Parameters
        ----------
        new_cfg : MutableMapping, optional
            the new configuration, by default None
        kwargs : Any, optional
            key value pairs, by default None

        """
        _new_cfg = new_cfg or CFG()
        if len(kwargs) > 0:  # avoid RecursionError
            _new_cfg.update(kwargs)
        for k in _new_cfg:
            # if _new_cfg[k].__class__.__name__ in ["dict", "EasyDict", "CFG"] and k in self:
            if isinstance(_new_cfg[k], MutableMapping) and k in self:
                self[k].update(_new_cfg[k])
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except Exception:
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """

        the updated pop method

        Parameters
        ----------
        key : str
            the key to pop
        default : Any, optional
            the default value, by default None

        """
        if key in self:
            delattr(self, key)
        return super().pop(key, default)


DEFAULTS = CFG()
DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)


def set_seed(random_seed: int) -> NoReturn:
    """
    set the seed of the random number generator

    Parameters
    ----------
    random_seed: int,
        the seed to be set

    """

    global DEFAULTS
    DEFAULTS.SEED = random_seed
    DEFAULTS.RNG = np.random.default_rng(seed=random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def execute_cmd(
    cmd: Union[str, List[str]],
    timeout_hour: Real = 6,
    quiet: bool = False,
    logger: Optional[Logger] = None,
    raise_error: bool = True,
    **kwargs: Any,
) -> Tuple[int, List[str]]:
    """ """
    shell_arg, executable_arg = kwargs.get("shell", False), kwargs.get(
        "executable", None
    )
    _cmd = shlex.split(cmd) if isinstance(cmd, str) else cmd
    s = subprocess.Popen(
        _cmd,
        shell=shell_arg,
        executable=executable_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=(not (platform.system().lower() == "windows")),
    )
    debug_stdout = collections.deque(maxlen=1000)
    msg = _str_center("  execute_cmd starts  ", 60)
    if logger:
        logger.info(msg)
    else:
        print(msg)
    start = time.time()
    now = time.time()
    timeout_sec = timeout_hour * 3600 if timeout_hour > 0 else float("inf")
    while now - start < timeout_sec:
        line = s.stdout.readline().decode("utf-8", errors="replace")
        if line.rstrip():
            debug_stdout.append(line)
            if logger:
                logger.debug(line)
            elif not quiet:
                print(line)
        exitcode = s.poll()
        if exitcode is not None:
            for line in s.stdout:
                debug_stdout.append(line.decode("utf-8", errors="replace"))
            if exitcode is not None and exitcode != 0:
                error_msg = " ".join(cmd) if not isinstance(cmd, str) else cmd
                error_msg += "\n"
                error_msg += "".join(debug_stdout)
                s.communicate()
                s.stdout.close()
                msg = _str_center("  execute_cmd failed  ", 60)
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                if raise_error:
                    raise subprocess.CalledProcessError(exitcode, error_msg)
                else:
                    output_msg = list(debug_stdout)
                    return exitcode, output_msg
            else:
                break
        now = time.time()
    # s.communicate()
    # s.terminate()
    s.kill()
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    # os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    s.stdout.close()
    output_msg = list(debug_stdout)

    msg = _str_center("  execute_cmd succeeded  ", 60)
    if logger:
        logger.info(msg)
    else:
        print(msg)

    exitcode = 0

    return exitcode, output_msg


def _str_center(s: str, length: int):
    """ """
    return s.center(length, "*").center(length + 2, "\n")


def dict_to_str(
    d: Union[dict, list, tuple], current_depth: int = 1, indent_spaces: int = 4
) -> str:
    """finished, checked,

    convert a (possibly) nested dict into a `str` of json-like formatted form,
    this nested dict might also contain lists or tuples of dict (and of str, int, etc.)

    Parameters
    ----------
    d: dict, or list, or tuple,
        a (possibly) nested `dict`, or a list of `dict`
    current_depth: int, default 1,
        depth of `d` in the (possible) parent `dict` or `list`
    indent_spaces: int, default 4,
        the indent spaces of each depth

    Returns
    -------
    s: str,
        the formatted string
    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = "{{}}" if isinstance(d, dict) else "[]"
        return s
    # flat_types = (Number, bool, str,)
    flat_types = (
        Number,
        bool,
    )
    flat_sep = ", "
    s = "\n"
    unit_indent = " " * indent_spaces
    prefix = unit_indent * current_depth
    if isinstance(d, (list, tuple)):
        if all([isinstance(v, flat_types) for v in d]):
            len_per_line = 110
            current_len = len(prefix) + 1  # + 1 for a comma
            val = []
            for idx, v in enumerate(d):
                add_v = f"\042{v}\042" if isinstance(v, str) else str(v)
                add_len = len(add_v) + len(flat_sep)
                if current_len + add_len > len_per_line:
                    val = ", ".join([item for item in val])
                    s += f"{prefix}{val},\n"
                    val = [add_v]
                    current_len = len(prefix) + 1 + len(add_v)
                else:
                    val.append(add_v)
                    current_len += add_len
            if len(val) > 0:
                val = ", ".join([item for item in val])
                s += f"{prefix}{val}\n"
        else:
            for idx, v in enumerate(d):
                if isinstance(v, (dict, list, tuple)):
                    s += f"{prefix}{dict_to_str(v, current_depth+1)}"
                else:
                    val = f"\042{v}\042" if isinstance(v, str) else v
                    s += f"{prefix}{val}"
                if idx < len(d) - 1:
                    s += ",\n"
                else:
                    s += "\n"
    elif isinstance(d, dict):
        for idx, (k, v) in enumerate(d.items()):
            key = f"\042{k}\042" if isinstance(k, str) else k
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{key}: {dict_to_str(v, current_depth+1)}"
            else:
                val = f"\042{v}\042" if isinstance(v, str) else v
                s += f"{prefix}{key}: {val}"
            if idx < len(d) - 1:
                s += ",\n"
            else:
                s += "\n"
    s += unit_indent * (current_depth - 1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s


def dicts_equal(d1: dict, d2: dict) -> bool:
    """finished, checked,

    Parameters
    ----------
    d1, d2: dict,
        the two dicts to compare equality

    Returns
    -------
    bool, True if `d1` equals `d2`

    NOTE
    ----
    the existence of numpy array, torch Tensor, pandas DataFrame and Series would probably
    cause errors when directly use the default `__eq__` method of dict,
    for example `{"a": np.array([1,2])} == {"a": np.array([1,2])}` would raise the following
    ```python
    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    ```

    Example
    -------
    >>> d1 = {"a": pd.DataFrame([{"hehe":1,"haha":2}])[["haha","hehe"]]}
    >>> d2 = {"a": pd.DataFrame([{"hehe":1,"haha":2}])[["hehe","haha"]]}
    >>> dicts_equal(d1, d2)
    ... True
    """
    if len(d1) != len(d2):
        return False
    for k, v in d1.items():
        if k not in d2 or not isinstance(d2[k], type(v)):
            return False
        if isinstance(v, dict):
            if not dicts_equal(v, d2[k]):
                return False
        elif isinstance(v, np.ndarray):
            if v.shape != d2[k].shape or not (v == d2[k]).all():
                return False
        elif isinstance(v, torch.Tensor):
            if v.shape != d2[k].shape or not (v == d2[k]).all().item():
                return False
        elif isinstance(v, pd.DataFrame):
            if v.shape != d2[k].shape or set(v.columns) != set(d2[k].columns):
                # consider: should one check index be equal?
                return False
            # for c in v.columns:
            #     if not (v[c] == d2[k][c]).all():
            #         return False
            if not (v.values == d2[k][v.columns].values).all():
                return False
        elif isinstance(v, pd.Series):
            if v.shape != d2[k].shape or v.name != d2[k].name:
                return False
            if not (v == d2[k]).all():
                return False
        # TODO: consider whether there are any other dtypes that should be treated similarly
        else:  # other dtypes whose equality can be directly checked
            if v != d2[k]:
                return False
    return True


def hashable(key: Any) -> bool:
    try:
        hash(key)
        return True
    except TypeError:
        return False


def consecutive_groups(iterable, ordering=lambda x: x):
    """Yield groups of consecutive items using :func:`itertools.groupby`.
    The *ordering* function determines whether two items are adjacent by
    returning their position.

    By default, the ordering function is the identity function. This is
    suitable for finding runs of numbers:

        >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
        >>> for group in consecutive_groups(iterable):
        ...     print(list(group))
        [1]
        [10, 11, 12]
        [20]
        [30, 31, 32, 33]
        [40]

    For finding runs of adjacent letters, try using the :meth:`index` method
    of a string of letters:

        >>> from string import ascii_lowercase
        >>> iterable = 'abcdfgilmnop'
        >>> ordering = ascii_lowercase.index
        >>> for group in consecutive_groups(iterable, ordering):
        ...     print(list(group))
        ['a', 'b', 'c', 'd']
        ['f', 'g']
        ['i']
        ['l', 'm', 'n', 'o', 'p']

    Each group of consecutive items is an iterator that shares it source with
    *iterable*. When an an output group is advanced, the previous group is
    no longer available unless its elements are copied (e.g., into a ``list``).

        >>> iterable = [1, 2, 11, 12, 21, 22]
        >>> saved_groups = []
        >>> for group in consecutive_groups(iterable):
        ...     saved_groups.append(list(group))  # Copy group elements
        >>> saved_groups
        [[1, 2], [11, 12], [21, 22]]

    """
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)


def dataframe_selection(
    df: pd.DataFrame, cols: Sequence[str], num: int
) -> pd.DataFrame:
    """
    perform uniform selection from `df` on `cols` with each combination randomly selected `num` rows
    """
    col_sets = product(*[set(df[c].tolist()) for c in cols])
    df_out = pd.DataFrame()
    for combination in col_sets:
        df_tmp = df.copy()
        for c, v in zip(cols, combination):
            df_tmp = df_tmp[df_tmp[c] == v]
        df_tmp = df_tmp.sample(n=min(num, len(df_tmp))).reset_index(drop=True)
        df_out = pd.concat([df_out, df_tmp], ignore_index=True)
    return df_out


def str2bool(v: Union[str, bool]) -> bool:
    """finished, checked,

    converts a "boolean" value possibly in the format of str to bool
    Parameters
    ----------
    v: str or bool,
        the "boolean" value
    Returns
    -------
    b: bool,
        `v` in the format of bool
    References
    ----------
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        b = v
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        b = True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        b = False
    else:
        raise ValueError("Boolean value expected.")
    return b


@contextmanager
def timeout(duration: float):
    """
    A context manager that raises a `TimeoutError` after a specified time.

    Parameters
    ----------
    duration: float,
        the time duration in seconds,
        should be non-negative,
        0 for no timeout

    References
    ----------
    https://stackoverflow.com/questions/492519/timeout-on-a-function-call

    """
    if np.isinf(duration):
        duration = 0
    elif duration < 0:
        raise ValueError("duration must be non-negative")
    elif duration > 0:  # granularity is 1 second, so round up
        duration = max(1, int(duration))

    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)
