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
    '''
    Abstract class for ihvp calculator
    '''
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
    '''
    Exact ihvp calculator.
    The calculator will directly calculate the hessian matrix and
    inverse the matrix for pre-calculation.
    '''
    def __init__(self,
                 func: Callable,
                 arg_num: int = 0) -> None:
        '''
        Create an ExactIHVP calculator.

        :param func: A Python function that takes one or more arguments.
               Must return a single-element Tensor. The hessian will
               be calculated on this function.
        :param arg_num: An integer default to 0. Specifies arguments to compute
               gradients with respect to. 
        '''
        super().__init__(func, arg_num)
        self.inverse_hessian_tensor = None

    def cache(self,
              *args,
              **kwargs) -> None:
        '''
        Pre-calculation of the IHVP calculator. ExactIHVP calculator will
        directly calculate the inverse hessian matrix explicitly and store for
        later vector product calculation.

        `cache` must be called once before any calling of `product`.

        :param *args: The positional argument of the function.
        :param **kwargs: The keyword argument of the function.
        '''
        hessian_tensor = hessian(self.func)(*args, **kwargs)
        self.inverse_hessian_tensor = torch.linalg.inv(hessian_tensor)

    def product(self,
                vec: Tensor) -> Tensor:
        '''
        Calculate the inverse hessian vector product.

        :param vec: A tensor to product on the inverse hessian.
        '''
        if self.inverse_hessian_tensor is None:
            raise IHVPUsageError("You need to call `cache` before `product`.")
        return self.inverse_hessian_tensor @ vec


class IHVPUsageError(Exception):
    '''
    The class to indicate that a mis-usage is carried out by users.
    '''
