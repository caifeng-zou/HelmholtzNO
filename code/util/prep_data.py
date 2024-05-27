import torch
import numpy as np

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


'''This is fast but only valid when the data set follows the frequency order'''
class standardizer_wrt_freq(object):
    def __init__(self, x, eps=eps, flag='input'):
        super(standardizer_wrt_freq, self).__init__()
        self.eps = eps
        self.means_x = torch.zeros(NF, x.size(-1))
        self.stds_x = torch.zeros(NF, x.size(-1))

        n = x.size(0) // NF
        for i in range(NF):
            bx = x[i:i+n*NF:NF]
            self.means_x[i] = torch.mean(bx, dim=(0, 1, 2, 3))
            self.stds_x[i] = torch.std(bx, dim=(0, 1, 2, 3))
        
        # imag for w = 0
        self.stds_x[self.stds_x==0] = torch.inf

        if flag == 'input':
            self.means_x[:, -1] = ws[freq_to_keep].mean()
            self.stds_x[:, -1] = ws[freq_to_keep].std()
    
    def encode(self, x):
        n = x.size(0) // NF
        for i in range(n):
            x[i*NF:(i+1)*NF] = (x[i*NF:(i+1)*NF] - self.means_x.view(self.means_x.size(0), 1, 1, 1, self.means_x.size(1))) / \
                (self.stds_x.view(self.means_x.size(0), 1, 1, 1, self.means_x.size(1)) + self.eps)
            
        return x

    def decode(self, x):
        n = x.size(0) // NF
        for i in range(n):
            x[i*NF:(i+1)*NF] = x[i*NF:(i+1)*NF] * (self.stds_x.view(self.means_x.size(0), 1, 1, 1, self.means_x.size(1)) + self.eps) + \
                self.means_x.view(self.means_x.size(0), 1, 1, 1, self.means_x.size(1))
        
        # imag for w = 0
        mask = torch.abs(x).eq(float('inf'))
        x.masked_fill_(mask, 0)
        mask = torch.abs(x).isnan()
        x.masked_fill_(mask, 0)
        
        return x

    def cpu(self):
        self.means_x = self.means_x.cpu()
        self.stds_x = self.stds_x.cpu()

class InputNormalizer(object): 
    def __init__(self, x, eps=0, nmax=None):
        super(InputNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        if nmax is not None:
            self.mean = torch.mean(x[:nmax], dim=(0, 1, 2, 3))
            self.std = torch.std(x[:nmax], dim=(0, 1, 2, 3))
        else:
            self.mean = torch.mean(x, dim=(0, 1, 2, 3))
            self.std = torch.std(x, dim=(0, 1, 2, 3))
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def return_to_time(data_in_freq, n_after_padding=T):
    '''
    Input size: (N*NF, S, S, S, 6), the last dimension is (Ux, Uy, Uz) (complex)
    Return: (N, S, S, S, T, 3), the last dimension is (Ux, Uy, Uz) (real)
    '''
    data_in_time = data_in_freq.view(-1, NF, data_in_freq.size(-4), data_in_freq.size(-3), data_in_freq.size(-2), data_in_freq.size(-1))  # (N, NF, S, S, S, Vel*Cplx)
    data_in_time = data_in_time.view(data_in_time.size(0), data_in_time.size(1), data_in_time.size(2), data_in_time.size(3), data_in_time.size(4), 3, 2)  # (N, NF, S, S, S, Vel, Cplx)
    data_in_time = data_in_time.permute(0, 2, 3, 4, 5, 1, 6).contiguous()  # (N, S, S, S, Vel, NF, Cplx)
    data_in_time = torch.view_as_complex(data_in_time)  # (N, S, S, S, Vel, NF)
    kept_freq = torch.zeros(data_in_time.size(0), 
                            data_in_time.size(1), 
                            data_in_time.size(2), 
                            data_in_time.size(3),
                            data_in_time.size(4),
                            len(freqs), dtype=torch.cfloat)
    kept_freq[:, :, :, :, :, freq_to_keep] = data_in_time[:, :, :, :, :, :]
    data_in_time = torch.fft.irfft(kept_freq, n=n_after_padding, dim=-1, norm='backward')  #(N, S, S, S, Vel, T)
    data_in_time = data_in_time[:, :, :, :, :, :T]
    data_in_time = data_in_time.permute(0, 1, 2, 3, 5, 4)
    return data_in_time

def src_on_grid(srcxyz):
    ix = (srcxyz[0] - xcoor).abs().argmin()
    iy = (srcxyz[1] - ycoor).abs().argmin()
    iz = (srcxyz[2] - zcoor).abs().argmin()
    return torch.tensor((ix, iy, iz))

def convert_to_freq_in(input_in_time, n_after_padding=T):
    '''
    Input size: (N, S, S, S, T, 3), the last dimension includes Vp, Vs, and real source wave in time
    Return: (N*NF, S, S, S, 5), the last dimension includes Vp, Vs, complex source wave in freq, and w 
    '''
    repeat = torch.ones(1, 1, 1, 1, NF, 1)
    Vp = input_in_time[:, :, :, :, 0, 0].view(-1, nx, ny, nz, 1, 1)
    Vs = input_in_time[:, :, :, :, 0, 1].view(-1, nx, ny, nz, 1, 1)
    Vp = Vp * repeat
    Vs = Vs * repeat
    src = input_in_time[:, :, :, :, :, 2]  # source
    window = torch.hann_window(src.size(4)).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(src.size(0), src.size(1), src.size(2), src.size(3), 1)
    src = torch.fft.rfft(src * window, 
                         dim=4, norm='backward', n=n_after_padding)
    src = src[:, :, :, :, freq_to_keep]  
    src = torch.view_as_real(src)  # (N, S, S, S, NF, 2)
    w = ws[freq_to_keep]
    w = w.view(1, 1, 1, 1, -1, 1).repeat(input_in_time.size(0), nx, ny, nz, 1, 1)  # (N, S, S, S, NF, 1)
    input_in_freq = torch.cat((Vp, Vs, src, w), dim=-1)  # (N, S, S, S, NF, 5)
    input_in_freq = input_in_freq.permute(0, 4, 1, 2, 3, 5)  # (N, NF, S, S, S, 5)
    input_in_freq = input_in_freq.flatten(0, 1)  # (N*NF, S, S, S, 5), repeated w[0] to w[-1]
    return input_in_freq

def convert_to_freq_out(output_in_time, n_after_padding=T):
    '''
    Input size: (N, S, S, S, T, 3), the last dimension is (Ux, Uy, Uz) (real)
    Return: (N*NF, S, S, S, 6), the last dimension is (Ux, Uy, Uz) (complex)
    '''
    window = torch.hann_window(output_in_time.size(-2)).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(output_in_time.size(0), 
                                                                                                                                 output_in_time.size(1), 
                                                                                                                                 output_in_time.size(2), 
                                                                                                                                 output_in_time.size(3), 
                                                                                                                                 1,
                                                                                                                                 output_in_time.size(-1))
    output_in_freq = torch.fft.rfft(output_in_time * window, 
                                    dim=-2, norm='backward', n=n_after_padding)  # Negative frequencies omitted 
    output_in_freq = output_in_freq[:, :, :, :, freq_to_keep, :]
    output_in_freq = torch.view_as_real(output_in_freq.permute(0, 1, 2, 3, 5, 4))  # view_as_real operates on the last dimension (N, S, S, S, 2, NF, 2c)
    output_in_freq = output_in_freq.permute(0, 5, 1, 2, 3, 4, 6)  # Move the frenquency domain forward (N, NF, S, S, S, 2, 2c)
    output_in_freq = output_in_freq.flatten(-2, -1)  # Make complex u in the channel dimension
    output_in_freq = output_in_freq.flatten(0, 1)  # Make the freq dimension at the batch location for parallelization
    return output_in_freq

def get_train_data(offsets_train, nstrain, datapath, stf):
    train_a = torch.zeros(nx, ny, nz, T, 3)
    train_u = torch.zeros(nx, ny, nz, T, 3)
    train_in = torch.zeros(nstrain*NF, nx, ny, nz, in_channels)
    train_out = torch.zeros(nstrain*NF, nx, ny, nz, 6)
    
    i = 0
    for iS in range(offsets_train, offsets_train + nstrain, 1):     
        train_a[:, :, :, :, 0] = torch.from_numpy(np.load(datapath + 'vp_S{}.npy'.format(iS))).view(nx, ny, nz, 1)
        train_a[:, :, :, :, 1] = torch.from_numpy(np.load(datapath + 'vs_S{}.npy'.format(iS))).view(nx, ny, nz, 1)
        srcxyz = torch.from_numpy(np.load(datapath + 'src_S{}.npy'.format(iS)))
        srcxyz = src_on_grid(srcxyz)
        spatial_func = torch.exp(-(xxx - srcxyz[0]) ** 2 / src_width ** 2) * torch.exp(-(yyy - srcxyz[1]) ** 2 / src_width ** 2) * torch.exp(-(zzz - srcxyz[2]) ** 2 / src_width ** 2)
        train_a[:, :, :, :, 2] = spatial_func.view(nx, ny, nz, 1) * stf.view(1, 1, 1, -1)
        train_u = torch.from_numpy(np.load(datapath + "u_S{}.npy".format(iS)))

        train_in[i*NF:(i+1)*NF] = convert_to_freq_in(train_a.unsqueeze(0))
        train_out[i*NF:(i+1)*NF] = convert_to_freq_out(train_u.unsqueeze(0))
        i = i + 1
            
    return train_in, train_out
