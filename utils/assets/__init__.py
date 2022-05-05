# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-18
@LastEditTime: 2022-03-19
"""

import zipfile
import pickle
import gzip
import json
import re
from pathlib import Path
from typing import Union, Optional, Sequence

from ..strings import normalize_language, LANGUAGE


_DIR_PATH = Path(__file__).absolute().parent


__all__ = [
    "fetch",
]


def fetch(name: str, **kwargs):
    """ """
    func = f"""_fetch_{re.sub("[_-]+dict", "", name.lower()).replace("-", "_")}"""
    retval = eval(f"{func}(**kwargs)")
    return retval


def _fetch_cilin() -> dict:
    """ """
    cilin_path = _DIR_PATH / "cilin_dict.zip"
    with zipfile.ZipFile(cilin_path, "r") as archive:
        cilin_dict = pickle.loads(archive.read("cilin_dict.pkl"))
    return cilin_dict


def _fetch_fyh() -> tuple:
    """ """
    fyh_path = _DIR_PATH / "fyh_dict.zip"
    with zipfile.ZipFile(fyh_path, "r") as archive:
        tra_dict = pickle.loads(archive.read("tra_dict.pkl"))
        var_dict = pickle.loads(archive.read("var_dict.pkl"))
        hot_dict = pickle.loads(archive.read("hot_dict.pkl"))
    return tra_dict, var_dict, hot_dict


def _fetch_stopwords(language: str) -> list:
    """ """
    stopwords_path = _DIR_PATH / "stopwords.json.gz"
    with gzip.open(stopwords_path, "rt") as gz_file:
        stopwords = json.load(gz_file)
    stopwords = stopwords[normalize_language(language).value]
    return stopwords


def _fetch_stopwords_zh() -> list:
    """ """
    return _fetch_stopwords("zh")


def _fetch_stopwords_en() -> list:
    """ """
    return _fetch_stopwords("en")


def _fetch_sim() -> dict:
    """ """
    sd_path = _DIR_PATH / "sim_dict.pkl"
    return pickle.loads(sd_path.read_bytes())


def _fetch_hownet_en() -> dict:
    """ """
    hc_path = _DIR_PATH / "hownet_en.zip"
    with zipfile.ZipFile(hc_path, "r") as archive:
        hc = pickle.loads(archive.read("hownet_candidate/hownet_candidate.pkl"))
    return hc


def _fetch_hownet_zh() -> dict:
    """ """
    hc_path = _DIR_PATH / "hownet_zh.json.gz"
    with gzip.open(hc_path, "rt", encoding="utf-8") as f:
        hc = json.load(f)
    return hc


def _fetch_hownet(language: Union[str, LANGUAGE]) -> dict:
    _lang = normalize_language(language)
    if _lang == LANGUAGE.ENGLISH:
        return _fetch_hownet_en()
    elif _lang == LANGUAGE.CHINESE:
        return _fetch_hownet_zh()


def _fetch_checklist(keys: Optional[Union[Sequence[str], str]] = None) -> dict:
    """ """
    checklist_path = _DIR_PATH / "checklist_subs.json.gz"
    with gzip.open(checklist_path, "rt", encoding="utf-8") as f:
        checklist_subs = json.load(f)
    if keys is not None:
        if isinstance(keys, str):
            return checklist_subs[keys.upper()]
        _keys = [k.upper() for k in keys]
        return {k: v for k, v in checklist_subs.items() if k in _keys}
    return checklist_subs


def _fetch_checklist_subs(keys: Optional[Union[Sequence[str], str]] = None) -> dict:
    """ """
    return _fetch_checklist(keys)


def _fetch_dces() -> dict:
    """ """
    dces_path = _DIR_PATH / "DCES.zip"
    with zipfile.ZipFile(dces_path, "r") as archive:
        descs = pickle.loads(archive.read("descs.pkl"))
        try:
            neigh = pickle.loads(archive.read("neigh.pkl"))
        except ModuleNotFoundError:
            print("failed to load DCES neighbor. Init from sklearn.")
            from sklearn.neighbors import NearestNeighbors

            neigh = NearestNeighbors(
                **{
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "euclidean",
                    "metric_params": None,
                    "n_jobs": None,
                    "n_neighbors": 5,
                    "p": 2,
                    "radius": 1.0,
                }
            )
        vec_colnames = pickle.loads(archive.read("vec_colnames.pkl"))
    ret = {
        "descs": descs,
        "neigh": neigh,
        "vec_colnames": vec_colnames,
    }
    return ret
