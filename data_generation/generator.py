import numpy as np
import gstools as gs

def generate_3DGRF_simple(nx, ny, nz, ncorr=32, sigma=2, seed=-1):
    '''
    Generate Gaussian random fields with same correlation length scale in x and y
    Check https://geostat-framework.readthedocs.io/projects/gstools/en/stable/ for more details
    n - the shape of the output array
    ncorr - correlation length in the unit of grids
    seed - if needed
    ''' 
    x, y, z = range(nx), range(ny), range(nz)
    model = gs.Gaussian(dim=3, var=sigma**2, len_scale=ncorr)
    if seed >= 0:
        srf = gs.SRF(model, seed=seed)
    else:
        srf = gs.SRF(model)

    dV = srf((x, y, z), mesh_type='structured')
    return dV

def generate_3DGRF_vK(nx, ny, nz, ncorr=8, sigma=10, seed=-1):
    '''
    Von-Karman type random field, function adapted from Jorge C. Castellanos (jccastellanos@google.com)
    nx, ny, nz - the shape of the output array
    ncorr - correlation length in the unit of grids
    sigma - standard deviation
    seed - if needed
    '''
    if seed >= 0:
        np.random.seed(seed)

    deltax, deltay, deltaz = 1 / nx, 1 / ny, 1 / nz
    ax, ay, az = ncorr / nx, ncorr / ny, ncorr / nz
    Nx, Ny, Nz = 2 * nx, 2 * ny, 2 * nz

    dP = 2 * np.random.rand(Nx, Ny, Nz) - 1
    Y2 = np.fft.fftn(dP)

    kx1 = np.mod(1 / 2 + (np.arange(0, Nx)) / Nx, 1) - 1 / 2
    kx = kx1 * (2 * np.pi / deltax)

    ky1 = np.mod(1 / 2 + (np.arange(0, Ny)) / Ny, 1) - 1 / 2
    ky = ky1 * (2 * np.pi / deltay)

    kz1 = np.mod(1 / 2 + (np.arange(0, Nz)) / Nz, 1) - 1 / 2
    kz = kz1 * (2 * np.pi / deltaz)

    [KX, KY, KZ] = np.meshgrid(kx, ky, kz, indexing='ij')

    kappa = 0.5
    K_sq = KX**2 * ax**2 + KY**2 * ay**2 + KZ**2 * az**2
    P_K = (ax * ay * az) / ((1 + K_sq)**(1 + kappa))

    Y2 = Y2 * np.sqrt(P_K)
    dP_New = np.fft.ifftn(Y2)

    test = np.real(dP_New[0:Nx, 0:Ny, 0:Nz])
    test = sigma / np.std(test.flatten()) * test

    Mid_Nx, Mid_Ny, Mid_Nz = int(np.floor(Nx/4)), int(np.floor(Ny/4)), int(np.floor(Nz/4))
    Nx, Ny, Nz = int(Nx/2), int(Ny/2), int(Nz/2)
    dV = test[Mid_Nx:Mid_Nx+Nx, Mid_Ny:Mid_Ny+Ny, Mid_Nz:Mid_Nz+Nz]

    return dV

def perturbation_vs(vs):
    return 3 * (1 + vs / 100)

def perturbation_vpvsr(vp_vs_ratio):
    return 1.732 * (1 + vp_vs_ratio / 100)

def get_vp(vs, vp_vs_ratio):
    return vs * vp_vs_ratio
