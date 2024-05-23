import torch

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