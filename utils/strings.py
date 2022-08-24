# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-08-04
@LastEditTime: 2022-05-13

字符串的一些基本操作
"""

import re
import string
from enum import Enum
from dataclasses import dataclass
from typing import Union, Optional, List


__all__ = [
    "isascii",
    "words_from_text",
    "color_text",
    "LANGUAGE",
    "normalize_language",
    "UNIVERSAL_POSTAG",
    "normalize_pos_tag",
    "NERTAG",
    "normalize_ner_tag",
    "LABEL_COLORS",
    "process_label_name",
    "color_from_label",
    "color_from_output",
    "deEmojify",
    "isChinese",
    "get_str_justify_len",
    "check_if_subword",
    "check_if_punctuations",
    "strip_BPE_artifacts",
    "default_class_repr",
    "ReprMixin",
    "default_time_fmt",
]


def isascii(s: str) -> bool:
    try:
        return s.isascii()
    except AttributeError:
        return all([ord(c) < 128 for c in s])


def words_from_text(s: str, words_to_ignore: list = []) -> list:
    """ """
    homos = "ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"
    exceptions = "'-_*@"
    filter_pattern = homos + "'\\-_\\*@"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words


class _ANSI_ESCAPE_CODES(Enum):
    """Escape codes for printing color to the terminal."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    GRAY = "\033[37m"
    PURPLE = "\033[35m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    """ This color stops the current color sequence. """
    STOP = "\033[0m"


def color_text(
    text: str, color: Optional[str] = None, method: Optional[str] = None
) -> str:
    if not (isinstance(color, str) or isinstance(color, tuple)):
        raise TypeError(f"Cannot color text with provided color of type {type(color)}")
    if isinstance(color, tuple):
        if len(color) > 1:
            text = color_text(text, color[1:], method)
        color = color[0]

    if method is None:
        return text
    if method == "html":
        return f"<font color = {color}>{text}</font>"
    elif method == "ansi":
        color = color.lower()
        if color == "green":
            color = _ANSI_ESCAPE_CODES.OKGREEN.value
        elif color == "red":
            color = _ANSI_ESCAPE_CODES.FAIL.value
        elif color == "blue":
            color = _ANSI_ESCAPE_CODES.OKBLUE.value
        elif color == "purple":
            color = _ANSI_ESCAPE_CODES.PURPLE.value
        elif color == "gray":
            color = _ANSI_ESCAPE_CODES.GRAY.value
        elif color == "bold":
            color = _ANSI_ESCAPE_CODES.BOLD.value
        elif color == "underline":
            color = _ANSI_ESCAPE_CODES.UNDERLINE.value
        elif color == "warning":
            color = _ANSI_ESCAPE_CODES.WARNING.value
        else:
            raise ValueError(f"unknown text color {color}")

        return color + text + _ANSI_ESCAPE_CODES.STOP.value
    elif method == "file":
        return "[[" + text + "]]"


class UNIVERSAL_POSTAG(Enum):
    """
    Universal (Coarse) POS tags

    References
    ----------
    1. https://github.com/jind11/TextFooler/blob/master/criteria.py#L32
    2. https://www.nltk.org/book/ch05.html#tab-universal-tagset
    """

    NOUN = "nouns"  # 名词
    VERB = "verbs"  # 动词
    ADJ = "adjectives"  # 形容词
    ADV = "adverbs"  # 副词
    PRON = "pronouns"  # 代词
    DET = "determiners_articles"  # 定冠词，不定冠词
    ADP = "prepositions_postpositions"  # 介词
    NUM = "numerals"  # 数词
    CONJ = "conjunctions"  # 连词
    PRT = "particles"  # 助词
    PNCT = "punctuations"  # 标点符号
    OTHER = "other"  # 其他


# fmt: off
_pos_tag_mapping = {
    UNIVERSAL_POSTAG.NOUN: [  # nouns
        "n", "f", "s", "t", "nr", "ns", "nt", "nw", "nz",  # jieba paddle
        "ng", "nrfg", "nrt", "tg",  # jieba ictclas
        "NN", "NNP", "NNPS", "NNS",  # nltk
    ],
    UNIVERSAL_POSTAG.VERB: [  # verbs
        "v", "vd", "vn",  # jieba paddle
        "vg", "vi", "vq",  # jieba ictclas
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"  # nltk
    ],
    UNIVERSAL_POSTAG.ADJ: [  # adjectives
        "a", "ad", "an",  # jieba paddle
        "ag",  # jieba ictclas
        "JJ", "JJR", "JJS",  # nltk
    ],
    UNIVERSAL_POSTAG.ADV: [  # adverbs
        "d",  # jieba paddle
        "df", "dg",  # jieba ictclas
        "RB", "RBR", "RBS", "WRB",  # nltk
    ],
    UNIVERSAL_POSTAG.PRON: [  # pronouns
        "r",  # jieba paddle
        "rg", "rr", "rz",  # jieba ictclas
        "PRP", "PRP$", "WP", "WP$",  # nltk
    ],
    UNIVERSAL_POSTAG.DET: [  # determiners and articles
        # jieba paddle
        "DT", "PDT", "WDT"  # nltk
    ],
    UNIVERSAL_POSTAG.ADP: [  # prepositions and postpositions
        "p",  # jieba paddle
        "IN",  # nltk
    ],
    UNIVERSAL_POSTAG.NUM: [  # numerals
        "m", "q",  # jieba paddle
        "mg", "mq",  # jieba ictclas
        "CD",  # nltk
    ],
    UNIVERSAL_POSTAG.CONJ: [  # conjunctions
        "c",  # jieba paddle
        "CC",  # nltk
    ],
    UNIVERSAL_POSTAG.PRT: [  # particles
        "u",  # jieba paddle
        "ud", "ug", "uj", "ul", "uv", "uz",  # jieba ictclas
        "RP",  # nltk
    ],
    UNIVERSAL_POSTAG.PNCT: [  # punctuation
        "w",  # jieba paddle
        "$", "''", "(", ")", ",", "--", ".", ":",  # nltk
    ],
    UNIVERSAL_POSTAG.OTHER: [  # other
        "xc",  # jieba paddle
        "b", "e", "g", "h", "i", "j", "k", "l", "o", "x", "y", "z", "zg",  # jieba ictclas
        "EX", "FW", "LS", "MD", "POS", "SYM", "TO", "UH", "X",  # nltk
    ],
}
# fmt: on
for k, v in _pos_tag_mapping.items():
    v.append(k.name)  # universal postag itself
_pos_tag_mapping = {item: k for k, v in _pos_tag_mapping.items() for item in v}


def normalize_pos_tag(
    pos_tag: Union[str, UNIVERSAL_POSTAG, type(None)]
) -> Union[UNIVERSAL_POSTAG, type(None)]:
    """ """
    if pos_tag is None or type(pos_tag).__name__ == UNIVERSAL_POSTAG.__name__:
        return pos_tag
    try:
        return _pos_tag_mapping[pos_tag]
    except Exception:
        raise ValueError(f"词性{pos_tag}未定义")


@dataclass
class NERTAG:
    """
    Named Entity Recognition tags
    """

    value: str
    score: float = 1.0


def normalize_ner_tag(
    ner_tag: Union[str, NERTAG, type(None)], score: float = 1.0
) -> Union[NERTAG, type(None)]:
    """ """
    if ner_tag is None or type(ner_tag).__name__ == NERTAG.__class__.__name__:
        return ner_tag
    return NERTAG(ner_tag, score)


class LANGUAGE(Enum):
    """
    ISO 639-1 codes
    https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
    """

    CHINESE = "zh"
    ENGLISH = "en"


def normalize_language(language: Union[str, LANGUAGE, type(None)]) -> LANGUAGE:
    """ """
    if language is None or type(language).__name__ == LANGUAGE.__name__:
        return language
    # fmt: off
    _lang = {
        LANGUAGE.CHINESE: [
            "zh", "zho", "chi", "cn", "chinese", "中文", "zh-cn",
        ],
        LANGUAGE.ENGLISH: [
            "en", "eng", "english", "英文", "en-us",
        ],
    }
    # fmt: on
    _lang = {item: k for k, v in _lang.items() for item in v}
    try:
        return _lang[language.lower()]
    except Exception:
        raise ValueError(f"暂时不支持{language}，请检查是否拼写错误")


class LABEL_COLORS(Enum):
    """ """

    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    YELLOW = "yellow"
    ORANGE = "orange"
    CYAN = "cyan"
    BROWN = "brown"
    GRAY = "gray"


def process_label_name(label_name: str) -> str:
    """Takes a label name from a dataset and makes it nice.

    Meant to correct different abbreviations and automatically
    capitalize.
    """
    label_name = label_name.lower()
    if label_name == "neg":
        label_name = "negative"
    elif label_name == "pos":
        label_name = "positive"
    return label_name.capitalize()


def color_from_label(label_num: int) -> str:
    """Arbitrary colors for different labels."""
    try:
        label_num %= len(LABEL_COLORS)
        return [c.value for c in LABEL_COLORS][label_num]
    except TypeError:
        return LABEL_COLORS.BLUE.value


def color_from_output(label_name: str, label: int) -> str:
    """Returns the correct color for a label name, like 'positive', 'medicine',
    or 'entailment'."""
    label_name = label_name.lower()
    if label_name in {"entailment", "positive"}:
        return LABEL_COLORS.GREEN.value
    elif label_name in {"contradiction", "negative"}:
        return LABEL_COLORS.RED.value
    elif label_name in {"neutral"}:
        return LABEL_COLORS.GRAY.value
    else:
        # if no color pre-stored for label name, return color corresponding to
        # the label number (so, even for unknown datasets, we can give each
        # class a distinct color)
        return color_from_label(label)


def deEmojify(text: str) -> str:
    regrex_pattern = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


def isChinese(text: str, strict: bool = False) -> bool:
    """ """
    pattern = "[\u4e00-\u9FFF]"
    if not re.search(pattern, text):  # contains Chinese
        return False
    if not strict:
        pattern = "^[\u4e00-\u9FFF0-9a-zA-Z]+$"  # allow for Chinese along with numbers and English characters
    else:
        pattern = "^[\u4e00-\u9FFF]+$"  # only Chinese
    if not re.search(pattern, text):
        return False
    return True


def get_str_justify_len(text: str, max_len: int) -> int:
    """ """
    return max_len - len(text.encode("GBK")) + len(text)


def check_if_subword(token: str, model_type: str, starting: bool = False) -> bool:
    """Check if ``token`` is a subword token that is not a standalone word.

    Args:
        token: token to check.
        model_type: type of model (options: "bert", "roberta", "xlnet", etc.).
        starting: Should be set ``True`` if this token is the starting token of the overall text.
            This matters because models like RoBERTa does not add "Ġ" to beginning token.
    Returns:
        ``True`` if ``token`` is a subword token.
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return True if "##" in token else False
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        if starting:
            return False
        else:
            return False if token[0] == "Ġ" else True
    elif model_type == "xlnet":
        return False if token[0] == "_" else True
    else:
        return False


def strip_BPE_artifacts(token: str, model_type: str) -> str:
    """Strip characters such as "Ġ" that are left over from BPE tokenization."""
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return token.replace("##", "")
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        return token.replace("Ġ", "")
    elif model_type == "xlnet":
        if len(token) > 1 and token[0] == "_":
            return token[1:]
        else:
            return token
    else:
        return token


def check_if_punctuations(word: str) -> bool:
    """Returns ``True`` if ``word`` is just a sequence of punctuations."""
    for c in word:
        if c not in string.punctuation:
            return False
    return True


def default_class_repr(c: object, align: str = "center", depth: int = 1) -> str:
    """finished, checked,

    Parameters
    ----------
    c: object,
        the object to be represented
    align: str, default "center",
        the alignment of the class arguments

    Returns
    -------
    str,
        the representation of the class
    """
    indent = 4 * depth * " "
    closing_indent = 4 * (depth - 1) * " "
    if not hasattr(c, "extra_repr_keys"):
        return repr(c)
    elif len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = (
            "(\n"
            + ",\n".join(
                [
                    f"""{indent}{k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {default_class_repr(eval(f"c.{k}"),align,depth+1)}"""
                    for k in c.__dir__()
                    if k in c.extra_repr_keys()
                ]
            )
            + f"{closing_indent}\n)"
        )
    else:
        extra_str = ""
    return f"{c.__class__.__name__}{extra_str}"


class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__.
    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """ """
        return []


default_time_fmt = "%Y-%m-%d %H:%M:%S"
