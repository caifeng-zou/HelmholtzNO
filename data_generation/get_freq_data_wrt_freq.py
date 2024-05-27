
import torch
import numpy as np
import sys
sys.path.append('../code')

from joblib import load
from multiprocessing import Pool
from util.prep_data import src_on_grid, convert_to_freq_in, convert_to_freq_out, standardizer_wrt_freq

datapath = '../data/'
stf = torch.from_numpy(np.load(datapath + "stf_S0.npy"))

eps = 0
fs = 20
fn = fs / 2  # Nyquist frequency
dt = 1 / fs
start_time_in_seconds = -0.5
end_time_in_seconds = 2.0
T = round((end_time_in_seconds - start_time_in_seconds) / dt + 1)
n_after_padding = T
freqs = torch.arange(n_after_padding // 2 + 1) * fs / (n_after_padding - 1)
ws = 2 * torch.pi * freqs
freq_to_keep = list(range(5, torch.where(freqs>=6)[0][0].item() + 1))
NF = len(freq_to_keep)

nx, ny, nz = 64, 64, 64
src_width = 2
in_channels = 5

xcoor, ycoor, zcoor = torch.linspace(0, 5000, nx), torch.linspace(0, 5000, ny), torch.linspace(0, 5000, nz)
xx, yy, zz = torch.meshgrid(xcoor, ycoor, zcoor, indexing='ij')
xxx, yyy, zzz= torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")


def process(chunk):
    for iS in chunk:
        a = torch.zeros(nx, ny, nz, T, 3)
        u = torch.zeros(nx, ny, nz, T, 3)
        freq_in = torch.zeros(NF, nx, ny, nz, in_channels)
        freq_out = torch.zeros(NF, nx, ny, nz, 6)
        freq_srcxyz = torch.zeros(3)
        
        a[:, :, :, :, 0] = torch.from_numpy(np.load(datapath + 'vp_S{}.npy'.format(iS))).view(nx, ny, nz, 1)
        a[:, :, :, :, 1] = torch.from_numpy(np.load(datapath + 'vs_S{}.npy'.format(iS))).view(nx, ny, nz, 1)
        srcxyz = torch.from_numpy(np.load(datapath + 'src_S{}.npy'.format(iS)))
        freq_srcxyz = srcxyz
        srcxyz = src_on_grid(srcxyz)
        spatial_func = torch.exp(-(xxx - srcxyz[0]) ** 2 / src_width ** 2) * torch.exp(-(yyy - srcxyz[1]) ** 2 / src_width ** 2) * torch.exp(-(zzz - srcxyz[2]) ** 2 / src_width ** 2)
        a[:, :, :, :, 2] = spatial_func.view(nx, ny, nz, 1) * stf.view(1, 1, 1, -1)
        u = torch.from_numpy(np.load(datapath + "u_S{}.npy".format(iS)))

        freq_in = convert_to_freq_in(a.unsqueeze(0))
        freq_out = convert_to_freq_out(u.unsqueeze(0))
        
        freq_in = x_standardizer.encode(freq_in)
        freq_out = y_standardizer.encode(freq_out)
        
        np.save(datapath+"input_S{}.npy".format(iS), freq_in)
        np.save(datapath+"output_S{}.npy".format(iS), freq_out)
        np.save(datapath+"src_S{}.npy".format(iS), freq_srcxyz)
        

x_standardizer = load("../model/x_standardizer_wrt_freq.sav")
y_standardizer = load("../model/y_standardizer_wrt_freq.sav")

nproc = 1
nchunk = 1 // nproc
chunks = [range(i*nchunk, (i+1)*nchunk) for i in range(nproc)]

with Pool(processes=nproc) as p:
    p.map(process, chunks)
    


    


