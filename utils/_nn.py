# !/usr/bin/env python
# coding=UTF-8
"""
@Author: WEN Hao
@LastEditors: WEN Hao
@Description:
@Date: 2021-09-12
@LastEditTime: 2021-09-12

模型（神经网络）需要用到的一些公共基础的模块
"""

from typing import Sequence, NoReturn, Tuple

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import log_softmax


__all__ = [
    "AdaptiveSoftmax",
    "AdaptiveLoss",
    "RNNModel",
]


class AdaptiveSoftmax(nn.Module):
    """ """

    __name__ = "AdaptiveSoftmax"

    def __init__(
        self, input_size: int, cutoffs: Sequence[int], scale_down: int = 4
    ) -> NoReturn:
        """ """
        super().__init__()
        self.input_size = input_size
        self.cutoffs = cutoffs
        self.output_size = cutoffs[0] + len(cutoffs) - 1
        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        for i in range(len(cutoffs) - 1):
            seq = nn.Sequential(
                nn.Linear(input_size, input_size // scale_down, False),
                nn.Linear(input_size // scale_down, cutoffs[i + 1] - cutoffs[i], False),
            )
            self.tail.append(seq)

    def reset(self, init: float = 0.1) -> NoReturn:
        """ """
        self.head.weight.data.uniform_(-init, init)
        for tail in self.tail:
            for layer in tail:
                layer.weight.data.uniform_(-init, init)

    def set_target(self, target: torch.Tensor) -> NoReturn:
        """ """
        self.id = []
        for i in range(len(self.cutoffs) - 1):
            mask = target.ge(self.cutoffs[i]).mul(target.lt(self.cutoffs[i + 1]))
            if mask.sum() > 0:
                self.id.append(Variable(mask.float().nonzero().squeeze(1)))
            else:
                self.id.append(None)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ """
        assert len(inp.size()) == 2
        output = [self.head(inp)]
        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(self.tail[i](inp.index_select(0, self.id[i])))
            else:
                output.append(None)
        return output

    def log_prob(self, inp: torch.Tensor) -> torch.Tensor:
        """ """
        assert len(inp.size()) == 2
        head_out = self.head(inp)
        n = inp.size(0)
        prob = torch.zeros(n, self.cutoffs[-1]).to(inp.device)
        lsm_head = log_softmax(head_out, dim=head_out.dim() - 1)
        prob.narrow(1, 0, self.output_size).add_(
            lsm_head.narrow(1, 0, self.output_size).data
        )
        for i in range(len(self.tail)):
            pos = self.cutoffs[i]
            i_size = self.cutoffs[i + 1] - pos
            buff = lsm_head.narrow(1, self.cutoffs[0] + i, 1)
            buff = buff.expand(n, i_size)
            temp = self.tail[i](inp)
            lsm_tail = log_softmax(temp, dim=temp.dim() - 1)
            prob.narrow(1, pos, i_size).copy_(buff.data).add_(lsm_tail.data)
        return prob


class AdaptiveLoss(nn.Module):
    """ """

    __name__ = "AdaptiveLoss"

    def __init__(self, cutoffs: Sequence[int]) -> NoReturn:
        """ """
        super().__init__()
        self.cutoffs = cutoffs
        self.criterions = nn.ModuleList()
        for i in self.cutoffs:
            self.criterions.append(nn.CrossEntropyLoss(size_average=False))

    def reset(self) -> NoReturn:
        for criterion in self.criterions:
            criterion.zero_grad()

    def remap_target(self, target: torch.Tensor) -> torch.Tensor:
        """ """
        new_target = [target.clone()]
        for i in range(len(self.cutoffs) - 1):
            mask = target.ge(self.cutoffs[i]).mul(target.lt(self.cutoffs[i + 1]))

            if mask.sum() > 0:
                new_target[0][mask] = self.cutoffs[0] + i
                new_target.append(target[mask].add(-self.cutoffs[i]))
            else:
                new_target.append(None)
        return new_target

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ """
        n = inp[0].size(0)
        target = self.remap_target(target.data)
        loss = 0
        for i in range(len(inp)):
            if inp[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= inp[i].size(1)
                criterion = self.criterions[i]
                loss += criterion(inp[i], Variable(target[i]))
        loss /= n
        return loss


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.

    Based on official pytorch examples
    """

    __name__ = "RNNModel"

    def __init__(
        self,
        rnn_type: str,
        ntoken: int,
        ninp: int,
        nhid: int,
        nlayers: int,
        cutoffs: Sequence[int],
        proj: bool = False,
        dropout: float = 0.5,
        tie_weights: bool = False,
        lm1b: bool = False,
    ) -> NoReturn:
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.lm1b = lm1b

        if rnn_type == "GRU":
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )

        self.proj = proj

        if ninp != nhid and proj:
            self.proj_layer = nn.Linear(nhid, ninp)

        # if tie_weights:
        #     if nhid != ninp and not proj:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder = nn.Linear(ninp, ntoken)
        #     self.decoder.weight = self.encoder.weight
        # else:
        #     if nhid != ninp and not proj:
        #         if not lm1b:
        #             self.decoder = nn.Linear(nhid, ntoken)
        #         else:
        #             self.decoder = adapt_loss
        #     else:
        #         self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        if proj:
            self.softmax = AdaptiveSoftmax(ninp, cutoffs)
        else:
            self.softmax = AdaptiveSoftmax(nhid, cutoffs)

        self.full = False

    def init_weights(self) -> NoReturn:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        if "proj" in vars(self):
            if self.proj:
                output = self.proj_layer(output)

        output = output.view(output.size(0) * output.size(1), output.size(2))

        if self.full:
            decode = self.softmax.log_prob(output)
        else:
            decode = self.softmax(output)

        return decode, hidden

    def init_hidden(self, bsz: int) -> Variable:
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
