import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()
        """
        Adapted from https://github.com/ashiq24/UNO/blob/main/integral_operators.py

        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        dim3 = Default output grid size along time t ( or 3rd dimension of output domain)
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = dim1 
            self.modes2 = dim2
            self.modes3 = dim3 // 2 + 1

        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        weights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1 = None,dim2=None,dim3=None):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3   

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm='forward')

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2+1, dtype=torch.cfloat, device=x.device)
            
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm='forward')
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3

        x_out = self.conv(x)

        # Avoid aliasing - Effective
        ft = torch.fft.rfftn(x_out, dim=[-3,-2,-1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        x_out = torch.nn.functional.interpolate(x_out, size=(dim1, dim2, dim3), mode='trilinear', align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3, Normalize=True, Non_Lin=True):
        super(OperatorBlock_3D, self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1, dim2, dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim), affine=True)


    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x, dim1, dim2, dim3)
        x2_out = self.w(x, dim1, dim2, dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)

        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
    
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(int(in_channels), int(mid_channels), 1)
        self.mlp2 = nn.Conv3d(int(mid_channels), int(out_channels), 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
def kernel(in_chan=2, out_chan=4, up_dim=16):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, out_chan, bias=False)
            )
    return layers

class UNO3D(nn.Module):
    def __init__(self, in_width, width, pad=0, factor=3/4, pad_both=False):  # original factor is 1
        super(UNO3D, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic
        self.pad_both = pad_both
        self.fc_n1 = nn.Linear(self.in_width, self.width//2)
        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)
        
        self.conv1 = OperatorBlock_3D(self.width, 2*factor*self.width, 96, 96, 48, 24, 24, 24)
        self.conv2 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 64, 64, 32, 16, 16, 16)
        self.conv3 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 32, 32, 16, 8, 8, 8)
        self.conv4 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 64, 64, 32, 8, 8, 8)
        self.conv5 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 96, 96, 48, 16, 16, 16)
        self.conv6 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 128, 128, 64, 24, 24, 24) 
        self.mlp1 = MLP(2*factor*self.width, 2*factor*self.width, 2*factor*self.width)
        self.mlp2 = MLP(4*factor*self.width, 4*factor*self.width, 4*factor*self.width)
        self.mlp3 = MLP(8*factor*self.width, 8*factor*self.width, 8*factor*self.width)
        self.mlp4 = MLP(4*factor*self.width, 4*factor*self.width, 4*factor*self.width)
        self.mlp5 = MLP(2*factor*self.width, 2*factor*self.width, 2*factor*self.width)
        self.mlp6 = MLP(2*self.width, 2*self.width, 2*self.width)
        
        self.fc1 = nn.Linear(6*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 6)
        self.knet = kernel(3*self.width+3, 3*self.width, 3*self.width)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc = self.fc_n1(x)
        x_fc = F.gelu(x_fc)
        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        x_c1 = self.conv1(x_fc0)
        x_c1 = self.mlp1(x_c1)
        x_c2 = self.conv2(x_c1)
        x_c2 = self.mlp2(x_c2)
        x_c3 = self.conv3(x_c2)
        x_c3 = self.mlp3(x_c3)
        x_c4 = self.conv4(x_c3)
        x_c4 = self.mlp4(x_c4)
        x_c4 = torch.cat([x_c4, x_c2], dim=1)
        x_c5 = self.conv5(x_c4)
        x_c5 = self.mlp5(x_c5)
        x_c5 = torch.cat([x_c5, x_c1], dim=1)
        x_c6 = self.conv6(x_c5)
        x_c6 = self.mlp6(x_c6)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        x_c6 = x_c6.permute(0, 2, 3, 4, 1)

        # L
        res3 = x_c6.shape[3]
        grid = self.get_grid(x_c6.shape, x_c6.device)
        kx = self.knet(torch.cat((x_c6, grid), dim=-1))
        x_out = torch.einsum('bxyzi,bxyzi->bxyi', kx, x_c6) / res3
        x_out = x_out.unsqueeze(3)
        
        # Q
        x_out = self.fc1(torch.cat((x_out, x_c6[:, :, :, [0], :]), dim=-1))
        x_out = F.gelu(x_out)
        x_out = self.fc2(x_out)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)