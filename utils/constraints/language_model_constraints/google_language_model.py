# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-12
@LastEditTime: 2022-03-19

Language Model的限制条件：Google Language Model
"""

import os
import sys
import random
from collections import defaultdict
from typing import NoReturn, Optional, Generator, Sequence, Iterator, List

import numpy as np
import tensorflow as tf
import google.protobuf as protobuf
import lru

try:
    tf.config.gpu.set_per_process_memory_growth(True)
except Exception:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from ..base import Constraint
from ...transformations import Transformation, WordSubstitute
from ...attacked_text import AttackedText
from ...misc import nlp_cache_dir
from ...strings import ReprMixin
from ..._download_data import download_if_needed


__all__ = [
    "GoogleLanguageModel",
    "GoogLMHelper",
    "LoadModel",
    "Vocabulary",
    "CharsVocabulary",
    "LM1BDataset",
]


_CACHE_DIR = os.path.join(nlp_cache_dir, "alzantot-goog-lm")


class GoogleLanguageModel(Constraint):
    """Constraint that uses the Google 1 Billion Words Language Model to
    determine the difference in perplexity between x and x_adv.

    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    """

    def __init__(
        self,
        top_n: Optional[int] = None,
        top_n_per_index: Optional[int] = None,
        compare_against_original: bool = True,
    ) -> NoReturn:
        """
        Args:
            top_n:
            top_n_per_index:
            compare_against_original:
                If `True`, compare new `x_adv` against the original `x`.
                Otherwise, compare it against the previous `x_adv`.
        """
        if not (top_n or top_n_per_index):
            raise ValueError(
                "Cannot instantiate GoogleLanguageModel without top_n or top_n_per_index"
            )
        self.lm = GoogLMHelper()
        self.top_n = top_n
        self.top_n_per_index = top_n_per_index
        super().__init__(compare_against_original)

    def check_compatibility(self, transformation: Transformation) -> bool:
        return isinstance(transformation, WordSubstitute)

    def _check_batch(
        self, transformed_texts: Sequence[AttackedText], reference_text: AttackedText
    ) -> List[AttackedText]:
        """Returns the `top_n` of transformed_texts, as evaluated by the language model."""
        if not len(transformed_texts):
            return []

        def get_probs(
            reference_text: AttackedText, transformed_texts
        ) -> List[np.ndarray]:
            word_subs_index = reference_text.first_word_diff_index(transformed_texts[0])
            if word_subs_index is None:
                return []

            prefix = reference_text.words[word_subs_index - 1]
            subs_words = np.array([t.words[word_subs_index] for t in transformed_texts])
            probs = self.lm.get_words_probs(prefix, subs_words)
            return probs

        # This creates a dictionary where each new key is initialized to [].
        word_subs_index_map = defaultdict(list)

        for idx, transformed_text in enumerate(transformed_texts):
            word_subs_index = reference_text.first_word_diff_index(transformed_text)
            word_subs_index_map[word_subs_index].append((idx, transformed_text))

        probs = []
        for word_subs_index, item_list in word_subs_index_map.items():
            # zip(*some_list) is the inverse operator of zip!
            item_indices, this_transformed_texts = zip(*item_list)
            # t1 = time.time()
            probs_of_subs_at_index = list(
                zip(item_indices, get_probs(reference_text, this_transformed_texts))
            )
            # Sort by probability in descending order and take the top n for this index.
            probs_of_subs_at_index.sort(key=lambda x: -x[1])
            if self.top_n_per_index:
                probs_of_subs_at_index = probs_of_subs_at_index[: self.top_n_per_index]
            probs.extend(probs_of_subs_at_index)
            # t2 = time.time()

        # Probs is a list of (index, prob) where index is the corresponding
        # position in transformed_texts.
        probs.sort(key=lambda x: x[0])

        # Now that they're in order, reduce to just a list of probabilities.
        probs = np.array(list(map(lambda x: x[1], probs)))

        # Get the indices of the maximum elements.
        max_el_indices = np.argsort(-probs)
        if self.top_n:
            max_el_indices = max_el_indices[: self.top_n]

        # Put indices in order, now, so that the examples are returned in the
        # same order they were passed in.
        max_el_indices.sort()

        return [transformed_texts[i] for i in max_el_indices]

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        return self._check_batch([transformed_text], reference_text)

    def extra_repr_keys(self) -> List[str]:
        return ["top_n", "top_n_per_index"] + super().extra_repr_keys()


class GoogLMHelper(ReprMixin):
    """An implementation of `<https://arxiv.org/abs/1804.07998>`_ adapted from
    `<https://github.com/nesl/nlp_adversarial_examples>`_.

    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    """

    __name__ = "GoogLMHelper"

    def __init__(self) -> NoReturn:
        """ """
        download_if_needed(
            uri="alzantot-goog-lm",
            source="aitesting",
            dst_dir=nlp_cache_dir,
            extract=True,
        )
        tf.get_logger().setLevel("INFO")
        self.PBTXT_PATH = os.path.join(_CACHE_DIR, "graph-2016-09-10-gpu.pbtxt")
        self.CKPT_PATH = os.path.join(_CACHE_DIR, "ckpt-*")
        self.VOCAB_PATH = os.path.join(_CACHE_DIR, "vocab-2016-09-10.txt")

        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50

        self.vocab = CharsVocabulary(self.VOCAB_PATH, self.MAX_WORD_LEN)
        with tf.device("/gpu:1"):
            self.graph = tf.Graph()
            self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            self.t = LoadModel(self.sess, self.graph, self.PBTXT_PATH, self.CKPT_PATH)

        self.lm_cache = lru.LRU(2**18)

    def clear_cache(self) -> NoReturn:
        self.lm_cache.clear()

    def get_words_probs_uncached(
        self, prefix_words: str, list_words: Sequence[str]
    ) -> np.ndarray:
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)

        if prefix_words.find("<S>") != 0:
            prefix_words = "<S> " + prefix_words
        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros(
            [self.BATCH_SIZE, self.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32
        )

        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        inputs = [[samples[-1]]]
        char_ids_inputs[0, 0, :] = char_ids_samples[-1]
        softmax = self.sess.run(
            self.t["softmax_out"],
            feed_dict={
                self.t["char_inputs_in"]: char_ids_inputs,
                self.t["inputs_in"]: inputs,
                self.t["targets_in"]: targets,
                self.t["target_weights_in"]: weights,
            },
        )
        words_ids = [self.vocab.word_to_id(w) for w in list_words]
        word_probs = [softmax[0][w_id] for w_id in words_ids]
        return np.array(word_probs)

    def get_words_probs(
        self, prefix: str, list_words: Sequence[str]
    ) -> List[np.ndarray]:
        """Retrieves the probability of words."""
        uncached_words = []
        for word in list_words:
            if (prefix, word) not in self.lm_cache:
                if word not in uncached_words:
                    uncached_words.append(word)
        probs = self.get_words_probs_uncached(prefix, uncached_words)
        for word, prob in zip(uncached_words, probs):
            self.lm_cache[prefix, word] = prob
        return [self.lm_cache[prefix, word] for word in list_words]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["lm_cache"] = self.lm_cache.get_size()
        return state

    def __setstate__(self, state) -> NoReturn:
        self.__dict__ = state
        self.lm_cache = lru.LRU(state["lm_cache"])


def LoadModel(
    sess: tf.compat.v1.Session, graph: tf.Graph, gd_file: str, ckpt_file: str
) -> dict:
    """Load the model from GraphDef and AttackCheckpoint.

    Args:
      gd_file: GraphDef proto text file.
      ckpt_file: TensorFlow AttackCheckpoint file.

    Returns:
      TensorFlow session and tensors dict.

    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    """
    tf.get_logger().setLevel("INFO")
    with graph.as_default():
        sys.stderr.write("Recovering graph.\n")
        with tf.io.gfile.GFile(gd_file) as f:
            s = f.read()
            gd = tf.compat.v1.GraphDef()
            protobuf.text_format.Merge(s, gd)

        tf.compat.v1.logging.info("Recovering Graph %s", gd_file)
        t = {}
        [
            t["states_init"],
            t["lstm/lstm_0/control_dependency"],
            t["lstm/lstm_1/control_dependency"],
            t["softmax_out"],
            t["class_ids_out"],
            t["class_weights_out"],
            t["log_perplexity_out"],
            t["inputs_in"],
            t["targets_in"],
            t["target_weights_in"],
            t["char_inputs_in"],
            t["all_embs"],
            t["softmax_weights"],
            t["global_step"],
        ] = tf.import_graph_def(
            gd,
            {},
            [
                "states_init",
                "lstm/lstm_0/control_dependency:0",
                "lstm/lstm_1/control_dependency:0",
                "softmax_out:0",
                "class_ids_out:0",
                "class_weights_out:0",
                "log_perplexity_out:0",
                "inputs_in:0",
                "targets_in:0",
                "target_weights_in:0",
                "char_inputs_in:0",
                "all_embs_out:0",
                "Reshape_3:0",
                "global_step:0",
            ],
            name="",
        )

        sys.stderr.write("Recovering checkpoint %s\n" % ckpt_file)
        sess.run("save/restore_all", {"save/Const:0": ckpt_file})
        sess.run(t["states_init"])

    return t


class Vocabulary(ReprMixin):
    """Class that holds a vocabulary for the dataset."""

    __name__ = "Vocabulary"

    def __init__(self, filename: str) -> NoReturn:
        """Initialize vocabulary.

        Args:
          filename (str): Vocabulary file name.
        """
        self.filename = filename
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with tf.io.gfile.GFile(self.filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == "<S>":
                    self._bos = idx
                elif word_name == "</S>":
                    self._eos = idx
                elif word_name == "UNK":
                    self._unk = idx
                if word_name == "!!!MAXTERMID":
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

    @property
    def bos(self) -> int:
        return self._bos

    @property
    def eos(self) -> int:
        return self._eos

    @property
    def unk(self) -> int:
        return self._unk

    @property
    def size(self) -> int:
        return len(self._id_to_word)

    def word_to_id(self, word: str) -> int:
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id: int) -> str:
        """Converts an ID to the word it represents.

        Args:
          cur_id: The ID

        Returns:
          The word that :obj:`cur_id` represents.
        """
        if cur_id < self.size:
            return self._id_to_word[cur_id]
        return "ERROR"

    def decode(self, cur_ids: Sequence[int]) -> str:
        """Convert a list of ids to a sentence, with space inserted."""
        return " ".join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence: str) -> np.ndarray:
        """Convert a sentence to a list of ids, with special tokens added."""
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)

    def extra_repr_keys(self) -> List[str]:
        return [
            "filename",
        ]


class CharsVocabulary(Vocabulary):
    """Vocabulary containing character-level information."""

    __name__ = "CharsVocabulary"

    def __init__(self, filename: str, max_word_length: int) -> NoReturn:
        """ """
        super().__init__(filename)
        self._max_word_length = max_word_length
        chars_set = set()

        for word in self._id_to_word:
            chars_set |= set(word)

        free_ids = []
        for i in range(256):
            if chr(i) in chars_set:
                continue
            free_ids.append(chr(i))

        if len(free_ids) < 5:
            raise ValueError("Not enough free char ids: %d" % len(free_ids))

        self.bos_char = free_ids[0]  # <begin sentence>
        self.eos_char = free_ids[1]  # <end sentence>
        self.bow_char = free_ids[2]  # <begin word>
        self.eow_char = free_ids[3]  # <end word>
        self.pad_char = free_ids[4]  # <padding>

        chars_set |= {
            self.bos_char,
            self.eos_char,
            self.bow_char,
            self.eow_char,
            self.pad_char,
        }

        self._char_set = chars_set
        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

        self.bos_chars = self._convert_word_to_char_ids(self.bos_char)
        self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

    @property
    def word_char_ids(self) -> np.ndarray:
        return self._word_char_ids

    @property
    def max_word_length(self) -> int:
        return self._max_word_length

    def _convert_word_to_char_ids(self, word: str) -> np.ndarray:
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = ord(self.pad_char)

        if len(word) > self.max_word_length - 2:
            word = word[: self.max_word_length - 2]
        cur_word = self.bow_char + word + self.eow_char
        for j in range(len(cur_word)):
            code[j] = ord(cur_word[j])
        return code

    def word_to_char_ids(self, word: str) -> np.ndarray:
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence: str) -> np.ndarray:
        chars_ids = [self.word_to_char_ids(cur_word) for cur_word in sentence.split()]
        return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


def get_batch(
    generator: Generator,
    batch_size: int,
    num_steps: int,
    max_word_length: int,
    pad: bool = False,
) -> tuple:
    """Read batches of input."""
    cur_stream = [None] * batch_size

    inputs = np.zeros([batch_size, num_steps], np.int32)
    char_inputs = np.zeros([batch_size, num_steps, max_word_length], np.int32)
    global_word_ids = np.zeros([batch_size, num_steps], np.int32)
    targets = np.zeros([batch_size, num_steps], np.int32)
    weights = np.ones([batch_size, num_steps], np.float32)

    no_more_data = False
    while True:
        inputs[:] = 0
        char_inputs[:] = 0
        global_word_ids[:] = 0
        targets[:] = 0
        weights[:] = 0.0

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        cur_stream[i] = list(generator.next())
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
                global_word_ids[i, cur_pos:next_pos] = cur_stream[i][2][:how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1 : how_many + 1]
                weights[i, cur_pos:next_pos] = 1.0

                cur_pos = next_pos
                cur_stream[i][0] = cur_stream[i][0][how_many:]
                cur_stream[i][1] = cur_stream[i][1][how_many:]
                cur_stream[i][2] = cur_stream[i][2][how_many:]

                if pad:
                    break

        if no_more_data and np.sum(weights) == 0:
            # There is no more data and this is an empty batch. Done!
            break
        yield inputs, char_inputs, global_word_ids, targets, weights


class LM1BDataset(ReprMixin):
    """Utility class for 1B word benchmark dataset.

    The current implementation reads the data from the tokenized text files.
    """

    __name__ = "LM1BDataset"

    def __init__(self, filepattern: str, vocab: Vocabulary) -> NoReturn:
        """Initialize LM1BDataset reader.

        Args:
          filepattern: Dataset file pattern.
          vocab: Vocabulary.
        """
        self._vocab = vocab
        self._all_shards = tf.io.gfile.glob(filepattern)
        tf.compat.v1.logging.info(
            "Found %d shards at %s", len(self._all_shards), filepattern
        )

    def _load_random_shard(self) -> Iterator:
        """Randomly select a file and read it."""
        return self._load_shard(random.choice(self._all_shards))

    def _load_shard(self, shard_name: str) -> Iterator:
        """Read one file and convert to ids.

        Args:
          shard_name: file path.

        Returns:
          list of (id, char_id, global_word_id) tuples.
        """
        tf.compat.v1.logging.info("Loading data from: %s", shard_name)
        with tf.io.gfile.GFile(shard_name) as f:
            sentences = f.readlines()
        chars_ids = [self.vocab.encode_chars(sentence) for sentence in sentences]
        ids = [self.vocab.encode(sentence) for sentence in sentences]

        global_word_ids = []
        current_idx = 0
        for word_ids in ids:
            current_size = len(word_ids) - 1  # without <BOS> symbol
            cur_ids = np.arange(current_idx, current_idx + current_size)
            global_word_ids.append(cur_ids)
            current_idx += current_size

        tf.compat.v1.logging.info("Loaded %d words.", current_idx)
        tf.compat.v1.logging.info("Finished loading")
        return zip(ids, chars_ids, global_word_ids)

    def _get_sentence(self, forever: bool = True) -> tuple:
        while True:
            ids = self._load_random_shard()
            for current_ids in ids:
                yield current_ids
            if not forever:
                break

    def get_batch(
        self, batch_size: int, num_steps: int, pad: bool = False, forever: bool = True
    ) -> tuple:
        return get_batch(
            self._get_sentence(forever),
            batch_size,
            num_steps,
            self.vocab.max_word_length,
            pad=pad,
        )

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab
