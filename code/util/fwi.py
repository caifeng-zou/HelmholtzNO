import torch 
import torch.nn as nn

nx, ny, nz = 64, 64, 64
device = torch.device('cuda')

class auto_diff_FWI(nn.Module):
    def __init__(self):
        super().__init__()
        self.vs = torch.nn.Parameter(torch.zeros(nx, ny, nz, requires_grad=True, dtype=torch.float))
        self.vp = torch.nn.Parameter(torch.zeros(nx, ny, nz, requires_grad=True, dtype=torch.float))
        
    def forward(self, x, model):
        x[..., 0] =  self.vp
        x[..., 1] =  self.vs
        out = model(x)
        return out

def laplacian_reg(param):
    assert param.shape == (nx, ny, nz)
    
    param1 = torch.cat((torch.full((1, ny, nz), float('nan'), device=device), param[:-1, :, :]), dim=0)
    param2 = torch.cat((param[1:, :, :], torch.full((1, ny, nz), float('nan'), device=device)), dim=0)
    param3 = torch.cat((torch.full((nx, 1, nz), float('nan'), device=device), param[:, :-1, :]), dim=1)
    param4 = torch.cat((param[:, 1:, :], torch.full((nx, 1, nz), float('nan'), device=device)), dim=1)
    param5 = torch.cat((torch.full((nx, ny, 1), float('nan'), device=device), param[:, :, :-1]), dim=2)
    param6 = torch.cat((param[:, :, 1:], torch.full((nx, ny, 1), float('nan'), device=device)), dim=2)
    
    neighbors = torch.stack((param1, param2, param3, param4, param5, param6), dim=0)
    avr = torch.nanmean(neighbors, dim=0)

    reg = torch.norm(param - avr, p=2) ** 2
    
    return reg