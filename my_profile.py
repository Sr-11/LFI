
# # from lfi.utils import *
import torch
import numpy as np
import cProfile 
from memory_profiler import profile
import timeit
import os

# # # @profile
# # def my_function():
# #     # Pdist2_(D,X,Y)
# #     Pdist2(X,Y)
# # # cProfile.run('Pdist2(X,Y)')
# # t = timeit.repeat(my_function, number=100, repeat=5)
# # print(t)

