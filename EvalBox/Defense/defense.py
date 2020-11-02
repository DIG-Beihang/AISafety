#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-04-09 13:55:10
@LastEditTime: 2019-04-12 11:03:05
"""


from abc import ABCMeta, abstractmethod


class Defense(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, device):
        """
        @description: 
        @param {
            model:
            device:
        } 
        @return: None
        """
        self.model = model
        self.device = device

    @abstractmethod
    def generate(self):

        """
        @description: Abstract method
        @param {type} 
        @return: 
        """
        raise NotImplementedError
