#!/usr/bin/env python

'''
ihvp (inverse hessian-vector product) calculation for an arbitrary function.
This module contains:
- 
'''

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor
from torch.func import hessian

class AbstractIHVP(ABC):

    @abstractmethod
    def __init__(self,
                 func: Callable,
                 arg_num: int = 0) -> None:
        '''
        Create a IHVP calculator.

        :param func: A Python function that takes one or more arguments.
               Must return a single-element Tensor. The hessian will
               be calculated on this function.
        :param arg_num: An integer default to 0. Specifies arguments to compute
               gradients with respect to. 
        '''
        self.func = func
        self.arg_num = arg_num
    
    @abstractmethod
    def cache(self,
              *args,
              **kwargs) -> None:
        '''
        Pre-calculation of the IHVP calculator. Different algorithm
        may have different pre-calculation process. Some will explicitly
        calculate the hessian matrix's inverse, some will prepare hvp or
        hessian function for later usage.

        :param *args: The positional argument of the function.
        :param **kwargs: The keyword argument of the function.
        '''

    @abstractmethod
    def product(self,
                vec: Tensor) -> Tensor:
        '''
        Calculate the inverse hessian vector product.

        :param vec: A tensor to product on the inverse hessian.
        '''


class ExactIHVP(AbstractIHVP):
    def __init__(self,
                 func: Callable,
                 arg_num: int = 0) -> None:
        super().__init__(func, arg_num)

    def cache(self,
              *args,
              **kwargs) -> None:
        hessian_tensor = hessian(self.func)(*args, **kwargs)
        print(hessian_tensor, hessian_tensor.shape)
        self.inverse_hessian_tensor = torch.linalg.inv(hessian_tensor)

    def product(self,
                vec: Tensor) -> Tensor:
        return self.inverse_hessian_tensor @ vec

