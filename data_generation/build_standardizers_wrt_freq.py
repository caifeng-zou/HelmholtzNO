import torch
import numpy as np
import sys
sys.path.append('../code')

from joblib import dump
from util.prep_data import get_train_data, standardizer_wrt_freq

datapath = '../data/'
stf = torch.from_numpy(np.load(datapath + "stf_S0.npy"))

train_in, train_out = get_train_data(offsets_train=0, nstrain=1, datapath=datapath, stf=stf)
x_standardizer = standardizer_wrt_freq(train_in, flag='input')
y_standardizer = standardizer_wrt_freq(train_out, flag='output')
#dump(x_standardizer, "../model/x_standardizer_wrt_freq.sav")
#dump(y_standardizer, "../model/y_standardizer_wrt_freq.sav")



