#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: kaggle.py
# @time: 2020/2/18 15:28

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/zy/my_git/practice/deep_learning/Dive_into_DL")
import d2lzh1981 as d2l
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
