# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-01
@LastEditTime: 2022-02-25

下载模型等数据，
download_from_aitesting 已测试
"""

import os
import shutil
import zipfile
import tempfile
import pathlib
from io import BytesIO
from urllib.request import urlopen
from typing import NoReturn, Optional, Tuple

import filelock
import tqdm
import requests

from .misc import nlp_cache_dir


__all__ = [
    "download_if_needed",
    "download",
    "download_from_openattack",
    "download_from_textattack",
    "download_from_aitesting",
]


_OPENATTACK_DOMAIN = "https://cdn.data.thunlp.org/TAADToolbox/"
_TEXTATTACK_DOMAIN = "https://textattack.s3.amazonaws.com/"
_AITESTING_DOMAIN = "http://218.245.5.12/NLP/"

_AITESTING_MODELS = [
    "EasternFantasyNoval-small.zip",
    "chinese-bert-wwm-ext.zip",
    "learning2write.zip",
    "gpt2.zip",
    "alzantot-goog-lm.zip",
    "distilroberta-base.zip",
    "crawl-300d-2M.vec.zip",
    "NEZHA-base.zip",
    "gpt2-medium-chinese.zip",
    "bert-base-uncased.zip",
    "bert_amazon_reviews_zh.zip",
    "cnn-sst.zip",
    "lstm-imdb.zip",
    "bert-base-chinese.zip",
    "chinese-merge-word-embedding.txt.zip",
    "roberta-base-finetuned-dianping-chinese.zip",
    "counter-fitted-vectors.txt.zip",
    "glove200.zip",
    "lstm-sst2.zip",
    "cnn-imdb.zip",
    "CPM-Generate-distill.zip",
    "infersent-encoder.zip",
    "roberta-base-finetuned-chinanews-chinese.zip",
    "roberta-base-finetuned-ifeng-chinese.zip",
    "roberta_sst.zip",
    "universal-sentence-encoder-multilingual_3.zip",
    "universal-sentence-encoder-xling-many_1.zip",
]
_AITESTING_MODELS = {k.split(".")[0]: k for k in _AITESTING_MODELS}

_AITESTING_DATASETS = [
    "jd_full_filtered.csv.gz",
    "dianping_filtered.csv.gz",
    "dianping_filtered_tiny.csv.gz",
    "dianping-train.csv.xz",
    "jd_binary_test.csv.xz",
    "imdb_reviews_full.csv.gz",
    "jd_full_test.csv.xz",
    "chinese_hownet_syn.json.gz",
    "jd_full_filtered_tiny.csv.gz",
    "dianping-test.csv.xz",
    "jd_binary_train.csv.xz",
    "imdb_reviews.csv.gz",
    "douban.dat",
    "sogou_news_csv.tar.gz",
    "jd_full_train.csv.xz",
    "jd_binary_filtered.csv.gz",
    "jd_binary_filtered_tiny.csv.gz",
]
_AITESTING_DATASETS = {k.split(".")[0]: k for k in _AITESTING_DATASETS}


def download_if_needed(
    uri: str,
    source: str = "aitesting",
    dst_dir: str = nlp_cache_dir,
    extract: bool = True,
) -> str:
    """ """
    dst, need_download = _format_dst(uri, dst_dir, extract)
    if not need_download:
        return dst
    return download(uri, source, dst_dir, extract)


def _format_dst(
    uri: str, dst_dir: str = nlp_cache_dir, extract: bool = True
) -> Tuple[str, bool]:
    """ """
    dst = os.path.join(dst_dir, *(uri.strip("/").split("/")))
    if not extract:
        if not dst.endswith(".zip"):
            dst = dst + ".zip"
        if os.path.exists(dst):
            return dst, False
        return dst, True
    else:
        if dst.endswith(".zip"):
            dst = dst.rstrip(".zip", "")
        if os.path.exists(dst):
            return dst, False
        else:
            os.makedirs(dst)
            return dst, True


def download(
    uri: str,
    source: str = "aitesting",
    dst_dir: str = nlp_cache_dir,
    extract: bool = True,
) -> str:
    """ """
    dst, _ = _format_dst(uri, dst_dir, extract)
    download_func = {
        "textattack": download_from_textattack,
        "openattack": download_from_openattack,
        "aitesting": download_from_aitesting,
    }[source]
    return download_func(uri, dst, extract)


def download_from_openattack(uri: str, dst: str, extract: bool = True) -> str:
    """ """
    url = _OPENATTACK_DOMAIN + uri.strip("/")
    print(f"downloading from {url} into {dst}")
    dst_dir = os.path.dirname(dst)
    dst_fn = dst + ".zip" if not dst.endswith(".zip") else dst
    os.makedirs(dst_dir, exist_ok=True)
    try:
        with urlopen(url) as f:
            zf = zipfile.ZipFile(BytesIO(f.read()))
    except OverflowError:
        with urlopen(url) as f:
            CHUNK_SIZE = 1024 * 1024 * 10
            ftmp = open(dst_fn, "wb")
            while True:
                data = f.read(CHUNK_SIZE)
                ftmp.write(data)
                if len(data) == 0:
                    break
            ftmp.flush()
            ftmp.close()
            zf = zipfile.ZipFile(dst_fn)
    if extract:
        os.makedirs(os.path.splitext(dst_fn)[0], exist_ok=True)
        zf.extractall(os.path.splitext(dst_fn)[0])
        ret_val = os.path.splitext(dst_fn)[0]
    else:
        zf.write(dst_fn)
        ret_val = dst_fn
    zf.close()
    return ret_val


def download_from_textattack(uri: str, dst: str, extract: bool = True) -> str:
    """ """
    url = _TEXTATTACK_DOMAIN + uri.strip("/")
    print(f"downloading from {url} into {dst}")
    dst_lock_path = dst + ".lock"
    file_lock = filelock.FileLock(dst_lock_path)
    file_lock.acquire()
    downloaded_file = tempfile.NamedTemporaryFile(dir=dst, suffix=".zip", delete=False)
    _http_get(url, downloaded_file)
    # Move or unzip the file.
    downloaded_file.close()
    if extract and zipfile.is_zipfile(downloaded_file.name):
        _unzip_file(downloaded_file.name, dst)
    else:
        dst = os.path.join(os.path.dirname(dst), os.path.basename(downloaded_file.name))
        print(f"Copying {downloaded_file.name} to {dst}.")
        shutil.copyfile(downloaded_file.name, dst)
    file_lock.release()
    # Remove the temporary file.
    os.remove(downloaded_file.name)
    return dst


def download_from_aitesting(uri: str, dst: str, extract: bool = True) -> str:
    """ """
    dst_dir = os.path.dirname(dst)
    dst_fn = dst + ".zip" if not dst.endswith(".zip") else dst
    os.makedirs(dst_dir, exist_ok=True)
    uri = uri.strip("/")
    if uri.endswith(".zip"):
        uri = uri.split(".")[0]
    if uri in _AITESTING_MODELS:
        url = f"{_AITESTING_DOMAIN}models/{_AITESTING_MODELS[uri]}"
    else:
        url = f"{_AITESTING_DOMAIN}datasets/{_AITESTING_DATASETS[uri]}"
    print(f"downloading from {url} into {dst_dir}")
    dst_lock_path = dst_fn + ".lock"
    file_lock = filelock.FileLock(dst_lock_path)
    file_lock.acquire()
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=dst_dir, suffix=".zip", delete=False
    )
    _http_get(url, downloaded_file)
    # Move or unzip the file.
    downloaded_file.close()
    if extract and zipfile.is_zipfile(downloaded_file.name):
        _unzip_file(downloaded_file.name, dst)
    else:
        print(f"Copying {downloaded_file.name} to {dst}.")
        shutil.copyfile(downloaded_file.name, dst)
    file_lock.release()
    # Remove the temporary file.
    os.remove(downloaded_file.name)
    return dst


def _http_get(url: str, out_file: str, proxies: Optional[dict] = None) -> NoReturn:
    """Get contents of a URL and save to a file.

    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    print(f"Downloading {url}.")
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403 or req.status_code == 404:
        raise Exception(f"Could not reach {url}.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            out_file.write(chunk)
    progress.close()


def _unzip_file(path_to_zip_file: str, unzipped_folder_path: str) -> NoReturn:
    """Unzips a .zip file to folder path."""
    print(f"Unzipping file {path_to_zip_file} to {unzipped_folder_path}.")
    ufp = pathlib.Path(unzipped_folder_path)
    with zipfile.ZipFile(path_to_zip_file) as zip_ref:
        if os.path.dirname(zip_ref.namelist()[0]).startswith(ufp.name):
            ufp = ufp.parent
        zip_ref.extractall(str(ufp))


# def _unzip_file(path_to_zip_file:str, unzipped_folder_path:str) -> NoReturn:
#     """Unzips a .zip file to folder path."""
#     print(f"Unzipping file {path_to_zip_file} to {unzipped_folder_path}.")
#     with zipfile.ZipFile(path_to_zip_file) as zip_ref:
#         for member in zip_ref.namelist():
#             filename = os.path.basename(member)
#             # skip directories
#             if not filename:
#                 continue
#             # copy file (taken from zipfile's extract)
#             source = zip_ref.open(member)
#             target = open(os.path.join(unzipped_folder_path, filename), "wb")
#             with source, target:
#                 shutil.copyfileobj(source, target)
