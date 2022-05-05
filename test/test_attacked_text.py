# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-09
@LastEditTime: 2021-10-04

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

from text.utils.attacked_text import AttackedText, ChineseAttackedText
from text.utils.strings_en import remove_space_before_punct  # noqa: F401


texts_zh = [
    "父亲节到了，给俩爹分别搞了个，第二天就到了，很迅速。晚上试用感觉不错，科技改变生活，以后遛弯跑步都方便了",
    "个人觉得很鸡肋，买的时候算是特价，三百不到，但我觉得很一般。三百多的水，没我想要的效果，还不如用用freeplus了！",
    "这本书的实体版是拿来收藏用的，无奈目前没地方收藏，就买了Kidle版的来看看，很让人失望，图片很小很不清晰，内容也没有什么新意，就是一些采访的对话，没有啥意思。我比较喜欢有大量精美图片的收藏版，毕竟衣服城堡什么的电视剧里很多时候都只有几个镜头，完全不够欣赏。",
    "应该是很不错的！早闻大名，正要拜读。书收到时没有包装我还吓了一跳，不过好在内里没什么问题，就是封面有点小卷角，应该是在运送途中磕碰到了……心疼，待我读完再来追加评论！",
    "我是看过了样张才下单的，否则根本不知道这套书的内容，很可能会和自己买过的单行本重复，所以，卖这种套装书（尤其是无对应的纸质版套装书）的时候，麻烦在商品页面完善商品信息，谢谢~ 另，这一个系列的书里都没有版权页，真的不知道让人怎么相信这是正版，虽然心里知道。",
]


texts_en = [
    "Flamboyant in some movies and artfully restrained in others, 65-year-old Jack Nicholson could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles with the precision of the insurance actuary.",
    "The nicest thing that can be said about Stealing Harvard (which might have been called Freddy Gets Molested by a Dog) is that it's not as obnoxious as Tom Green's Freddie Got Fingered.",
    "Not counting a few gross-out comedies I've been trying to forget, this is the first film in a long time that made me want to bolt the theater in the first 10 minutes.",
    "So purely enjoyable that you might not even notice it's a fairly straightforward remake of Hollywood comedies such as Father of the Bride.",
    "Disappointingly, the characters are too strange and dysfunctional, Tom included, to ever get under the skin, but this is compensated in large part by the off-the-wall dialogue, visual playfulness and the outlandishness of the idea itself.",
]


class TestAttackedText:
    def __init__(self):
        self.en_no = 0
        self.zh_no = 0
        self.at_en = AttackedText("en", texts_en[self.en_no])
        self.at_zh = ChineseAttackedText(texts_zh[self.zh_no])

    def test_words(self):
        # fmt: off
        assert self.at_en.words == [
            "Flamboyant", "in", "some", "movies", "and", "artfully", "restrained", "in", "others",
            "65-year-old", "Jack", "Nicholson", "could", "be", "looking", "at", "his", "12th", "Oscar",
            "nomination", "by", "proving", "that", "he's", "now", "more", "than", "ever", "choosing",
            "his", "roles", "with", "the", "precision", "of", "the", "insurance", "actuary"
        ]
        assert self.at_zh.words == [
            "父亲节", "到", "了", "给", "俩", "爹", "分别", "搞了个",
            "第二天", "就", "到", "了", "很", "迅速", "晚上", "试用",
            "感觉", "不错", "科技", "改变", "生活", "以后", "遛弯",
            "跑步", "都", "方便", "了"
        ]
        # fmt: on

    def test_window_around_index(self):
        assert self.at_en.text_window_around_index(5, 1) == "artfully"
        assert self.at_en.text_window_around_index(5, 2) == "and artfully"
        assert self.at_en.text_window_around_index(5, 3) == "and artfully restrained"
        assert (
            self.at_en.text_window_around_index(5, 4)
            == "movies and artfully restrained"
        )
        assert (
            self.at_en.text_window_around_index(5, 5)
            == "movies and artfully restrained in"
        )
        assert (
            self.at_en.text_window_around_index(5, float("inf")) + "."
            == texts_en[self.en_no]
        )
        assert self.at_zh.text_window_around_index(5, 1) == "爹"
        assert self.at_zh.text_window_around_index(5, 2) == "俩爹"
        assert self.at_zh.text_window_around_index(5, 3) == "俩爹分别"
        assert self.at_zh.text_window_around_index(5, 4) == "给俩爹分别"
        assert self.at_zh.text_window_around_index(5, 5) == "给俩爹分别搞了个"
        assert (
            self.at_zh.text_window_around_index(5, float("inf")) == texts_zh[self.zh_no]
        )

    def test_big_window_around_index(self):
        assert (
            self.at_en.text_window_around_index(0, 10**5) + "."
        ) == self.at_en.text
        assert (self.at_zh.text_window_around_index(0, 10**5)) == self.at_zh.text

    def test_window_around_index_start(self):
        assert self.at_en.text_window_around_index(0, 3) == "Flamboyant in some"
        assert self.at_zh.text_window_around_index(0, 3) == "父亲节到了"

    def test_window_around_index_end(self):
        assert self.at_en.text_window_around_index(38, 3) == "the insurance actuary"
        assert self.at_zh.text_window_around_index(26, 3) == "都方便了"

    def test_text(self):
        assert self.at_en.text == texts_en[self.en_no]
        assert self.at_zh.text == texts_zh[self.zh_no]

    def test_printable_text(self):
        assert self.at_en.printable_text() == texts_en[self.en_no]
        assert self.at_zh.printable_text() == texts_zh[self.zh_no]

    def test_tokenizer_input(self):
        assert self.at_en.tokenizer_input == texts_en[self.en_no]
        assert self.at_zh.tokenizer_input == texts_zh[self.zh_no]

    def test_word_replacement(self):
        assert (
            self.at_en.replace_word_at_index(3, "films").text
            == "Flamboyant in some films and artfully restrained in others, 65-year-old Jack Nicholson could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles with the precision of the insurance actuary."
        )
        assert (
            self.at_zh.replace_word_at_index(12, "非常").text
            == "父亲节到了，给俩爹分别搞了个，第二天就到了，非常迅速。晚上试用感觉不错，科技改变生活，以后遛弯跑步都方便了"
        )

    def test_multi_word_replacement(self):
        new_text = self.at_en.replace_words_at_indices(
            (1, 3, 12, 20, 24), ("inside", "films", "might", "via", "currently")
        )
        assert (
            new_text.text
            == "Flamboyant inside some films and artfully restrained in others, 65-year-old Jack Nicholson might be looking at his 12th Oscar nomination via proving that he's currently, more than ever, choosing his roles with the precision of the insurance actuary."
        )
        new_text = self.at_zh.replace_words_at_indices(
            (1, 6, 12, 16, 22), ("来", "都", "非常", "觉得", "散步")
        )
        assert new_text.text == "父亲节来了，给俩爹都搞了个，第二天就到了，非常迅速。晚上试用觉得不错，科技改变生活，以后散步跑步都方便了"

    def test_word_deletion(self):
        new_text = self.at_en.delete_word_at_index(2).delete_word_at_index(10)
        assert (
            new_text.text
            == "Flamboyant in movies and artfully restrained in others, 65-year-old Jack could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles with the precision of the insurance actuary."
        )
        new_text = (
            new_text.delete_word_at_index(0)
            .delete_word_at_index(0)
            .delete_word_at_index(0)
        )
        assert (
            new_text.text
            == "and artfully restrained in others, 65-year-old Jack could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles with the precision of the insurance actuary."
        )
        new_text = self.at_zh.delete_word_at_index(2).delete_word_at_index(10)
        assert new_text.text == "父亲节到，给俩爹分别搞了个，第二天就到，很迅速。晚上试用感觉不错，科技改变生活，以后遛弯跑步都方便了"
        new_text = (
            new_text.delete_word_at_index(0)
            .delete_word_at_index(0)
            .delete_word_at_index(0)
        )
        assert new_text.text == "，俩爹分别搞了个，第二天就到，很迅速。晚上试用感觉不错，科技改变生活，以后遛弯跑步都方便了"

    def test_word_insertion(self):
        new_text = self.at_en.insert_text_before_word_index(3, "famous and popular")
        assert (
            new_text.text
            == "Flamboyant in some famous and popular movies and artfully restrained in others, 65-year-old Jack Nicholson could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles with the precision of the insurance actuary."
        )
        new_text = new_text.insert_text_after_word_index(33, "on his own")
        assert (
            new_text.text
            == "Flamboyant in some famous and popular movies and artfully restrained in others, 65-year-old Jack Nicholson could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles on his own with the precision of the insurance actuary."
        )
        new_text = self.at_zh.insert_text_before_word_index(1, "马上")
        assert new_text.text == "父亲节马上到了，给俩爹分别搞了个，第二天就到了，很迅速。晚上试用感觉不错，科技改变生活，以后遛弯跑步都方便了"
        new_text = new_text.insert_text_after_word_index(18, "非常棒")
        assert (
            new_text.text == "父亲节马上到了，给俩爹分别搞了个，第二天就到了，很迅速。晚上试用感觉不错非常棒，科技改变生活，以后遛弯跑步都方便了"
        )

    def test_modified_indices(self):
        new_text = self.at_en.insert_text_after_word_index(
            2, "famous and popular"
        ).insert_text_after_word_index(33, "on his own")
        assert (
            new_text.text
            == "Flamboyant in some famous and popular movies and artfully restrained in others, 65-year-old Jack Nicholson could be looking at his 12th Oscar nomination by proving that he's now, more than ever, choosing his roles on his own with the precision of the insurance actuary."
        )
        for old_idx, new_idx in enumerate(new_text.attack_attrs["original_index_map"]):
            assert (self.at_en.words[old_idx] == new_text.words[new_idx]) or (
                new_idx == -1
            )
        new_text = (
            new_text.delete_word_at_index(0)
            .delete_word_at_index(15)
            .delete_word_at_index(15)
            .delete_word_at_index(15)
            .delete_word_at_index(20)
        )
        for old_idx, new_idx in enumerate(new_text.attack_attrs["original_index_map"]):
            assert (self.at_en.words[old_idx] == new_text.words[new_idx]) or (
                new_idx == -1
            )
        assert (
            new_text.text
            == "in some famous and popular movies and artfully restrained in others, 65-year-old Jack Nicholson could his 12th Oscar nomination by that he's now, more than ever, choosing his roles on his own with the precision of the insurance actuary."
        )
        new_text = self.at_zh.insert_text_after_word_index(
            0, "马上"
        ).insert_text_after_word_index(18, "非常棒")
        assert (
            new_text.text == "父亲节马上到了，给俩爹分别搞了个，第二天就到了，很迅速。晚上试用感觉不错非常棒，科技改变生活，以后遛弯跑步都方便了"
        )
        for old_idx, new_idx in enumerate(new_text.attack_attrs["original_index_map"]):
            assert (self.at_zh.words[old_idx] == new_text.words[new_idx]) or (
                new_idx == -1
            )
        new_text = (
            new_text.delete_word_at_index(0)
            .delete_word_at_index(15)
            .delete_word_at_index(15)
            .delete_word_at_index(15)
            .delete_word_at_index(20)
        )
        for old_idx, new_idx in enumerate(new_text.attack_attrs["original_index_map"]):
            assert (self.at_zh.words[old_idx] == new_text.words[new_idx]) or (
                new_idx == -1
            )
        assert new_text.text == "马上到了，给俩爹分别搞了个，第二天就到了，很迅速。晚上非常棒，科技改变生活，遛弯跑步都方便了"
