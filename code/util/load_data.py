import torch
import numpy as np

def get_chunk_data(offset, chunk_size, NF, nx, ny, nz, in_channels, out_channels):
    data_in = torch.empty(chunk_size*NF, nx, ny, nz, in_channels)
    data_out = torch.empty(chunk_size*NF, nx, ny, nz, out_channels)
    
    i = 0
    for iS in range(offset, offset + chunk_size, 1):     
        data_in[i*NF:(i+1)*NF] = torch.from_numpy(np.load("/net/ghisallo/scratch1/czou/data_3D_freq/input_S{}.npy".format(iS)))
        data_out[i*NF:(i+1)*NF] = torch.from_numpy(np.load("/net/ghisallo/scratch1/czou/data_3D_freq/output_S{}.npy".format(iS)))
        i = i + 1
        
    return data_in, data_out