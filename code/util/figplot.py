import torch
import numpy as np
import matplotlib.pyplot as plt

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
xcoor, ycoor, zcoor = torch.linspace(0, 5000, nx), torch.linspace(0, 5000, ny), torch.linspace(0, 5000, nz)
xx, yy, zz = torch.meshgrid(xcoor, ycoor, zcoor, indexing='ij')


def plot_3D(fig, X, Y, Z, data, cmap, iplot, title, vmin, vmax, extend, ticks=[2.9, 3.2, 3.5], if_clb=False):
    kw = {'cmap': cmap,
          'levels': np.linspace(vmin, vmax, 100),
          'extend': extend}
    
    ax = fig.add_subplot(iplot[0], iplot[1], iplot[2], projection='3d', facecolor='white')
        
    ax.set_title(title, fontsize=15)
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, **kw
    )
    _ = ax.contourf(
        X[:, 0, :], data[:, 0, :], Z[:, 0, :],
        zdir='y', offset=0, **kw
    )
    C = ax.contourf(
        data[0, :, :], Y[0, :, :], Z[0, :, :],
        zdir='x', offset=0, **kw
    )
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], zmin, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], zmin, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], zmax, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
    ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], zmin, **edges_kw)
    ax.plot([xmin, xmax], [ymax, ymax], zmin, **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    
    ax.axis('off')
    ax.set_aspect('equal')
    ax.invert_zaxis()
    if if_clb:
        clb = plt.colorbar(C, ax=ax, fraction=0.025, pad=0.001)
        clb.ax.tick_params(labelsize=15)
        clb.set_ticks(ticks)
    
    ax.view_init(azim=225)

def scatter_3D(fig, X, Y, Z, data, cmap, iplot, title, vmin, vmax, num_ticks=3, if_clb=False):
    kw = {'cmap': cmap,
          's': 1,
          'vmin': vmin,
          'vmax': vmax}
    
    ax = fig.add_subplot(iplot[0], iplot[1], iplot[2], projection='3d', facecolor='white')
        
    ax.set_title(title, fontsize=15)
    
    Xsca = np.hstack((X[:, :, X.shape[-1]//2].flatten(), X[:, X.shape[1]//2, :].flatten(), X[X.shape[0]//2, :, :].flatten()))
    Ysca = np.hstack((Y[:, :, Y.shape[-1]//2].flatten(), Y[:, Y.shape[1]//2, :].flatten(), Y[Y.shape[0]//2, :, :].flatten()))
    Zsca = np.hstack((Z[:, :, Z.shape[-1]//2].flatten(), Z[:, Z.shape[1]//2, :].flatten(), Z[Z.shape[0]//2, :, :].flatten()))
    datasca = np.hstack((data[:, :, X.shape[-1]//2].flatten(), data[:, X.shape[1]//2, :].flatten(), data[X.shape[0]//2, :, :].flatten()))
    C = ax.scatter(
        Xsca, Ysca, Zsca, c=datasca, **kw
    )
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3, linestyle='--')
    ax.plot([(xmin+xmax)/2, (xmin+xmax)/2], [(ymin+ymax)/2, (ymin+ymax)/2], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmax], [(ymin+ymax)/2, (ymin+ymax)/2], [(zmin+zmax)/2, (zmin+zmax)/2], **edges_kw)
    ax.plot([(xmin+xmax)/2, (xmin+xmax)/2], [ymin, ymax], [(zmin+zmax)/2, (zmin+zmax)/2], **edges_kw)
    
    ax.axis('off')
    ax.set_aspect('equal')
    ax.invert_zaxis()
    if if_clb:
        clb = plt.colorbar(C, ax=ax, fraction=0.025, pad=0.001)
        clb.ax.tick_params(labelsize=15)
        clb.locator = plt.MaxNLocator(num_ticks)
        clb.update_ticks()
    
    ax.view_init(azim=225)

def plot_f_3D(srcxyz, vs, vp_vs_ratio, true, pred, realorimag):
    vs = vs / 1000
    vp = vs * vp_vs_ratio
    misfit = pred - true
    
    fig = plt.figure(figsize=(36, 17), facecolor='white')
    fig.tight_layout()
    fig.facecolor = 'white'
    plt.rcParams['axes.facecolor'] = 'w'

    ax0 = fig.add_subplot(494, projection='3d', facecolor='white')
    ax0.plot([srcxyz[0]/1000, 5], [srcxyz[1]/1000, srcxyz[1]/1000],[srcxyz[2]/1000, srcxyz[2]/1000], color='gray', linestyle='--')
    ax0.plot([srcxyz[0]/1000, srcxyz[0]/1000], [5, srcxyz[1]/1000], [srcxyz[2]/1000, srcxyz[2]/1000], color='gray', linestyle='--')
    ax0.plot([srcxyz[0]/1000, srcxyz[0]/1000], [srcxyz[1]/1000, srcxyz[1]/1000], [5, srcxyz[2]/1000], color='gray', linestyle='--')
    ax0.plot(srcxyz[0]/1000, srcxyz[1]/1000, srcxyz[2]/1000, 'r*', markersize=20)
    ax0.set_xlabel("X (km)", fontsize=12)
    ax0.set_ylabel("Y (km)", fontsize=12)
    ax0.tick_params(labelsize=12)
    ax0.set_title("$\mathbf{Source}$", fontsize=15)
    ax0.set_xlim(0, 5)
    ax0.set_ylim(0, 5)
    ax0.set_zlim(0, 5)
    ax0.set_box_aspect([1, 1, 1])
    ax0.invert_zaxis()
    ax0.view_init(azim=225)
    ax0.text2D(0, 0.9, "Z (km)", transform=ax0.transAxes,
        ha='left', va='top', fontsize=12)
    
    plot_3D(fig, xx, yy, zz, vs, 'Spectral', (4, 9, 5), "$\mathbf{V_S (km/s)}$", np.percentile(vs, 1), np.percentile(vs, 99), 'both', if_clb=True)
    scatter_3D(fig, xx, yy, zz, vp, 'Spectral', (4, 9, 6), "$\mathbf{V_P (km/s)}$", np.percentile(vp, 1), np.percentile(vp, 99), 3, True)
    
    iplot = 10
    for step in [0, 5, 10]:
        vmax = np.percentile(true[step], 99)

        if realorimag == 'real':
            datagroup = [true[step, :, :, :, 0], pred[step, :, :, :, 0], misfit[step, :, :, :, 0], 
                         true[step, :, :, :, 2], pred[step, :, :, :, 2], misfit[step, :, :, :, 2],
                         true[step, :, :, :, 4], pred[step, :, :, :, 4], misfit[step, :, :, :, 4]]
        if realorimag == 'imag':
            datagroup = [true[step, :, :, :, 1], pred[step, :, :, :, 1], misfit[step, :, :, :, 1], 
                         true[step, :, :, :, 3], pred[step, :, :, :, 3], misfit[step, :, :, :, 3],
                         true[step, :, :, :, 5], pred[step, :, :, :, 5], misfit[step, :, :, :, 5]]
        
        namegroup = ["True $\mathregular{U_x}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "Predicted $\mathregular{U_x}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "Misfit $\mathregular{U_x}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "True $\mathregular{U_y}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "Predicted $\mathregular{U_y}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "Misfit $\mathregular{U_y}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "True $\mathregular{U_z}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "Predicted $\mathregular{U_z}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step]),
                     "Misfit $\mathregular{U_z}$ " + "{} of $\mathbf{{{:.4g}}}$ $\mathbf{{Hz}}$".format(realorimag, freqs[freq_to_keep][step])]
        
        for data, name in zip(datagroup, namegroup):
            plot_3D(fig, xx, yy, zz, data, 'PuOr', (4, 9, iplot), name, -vmax, vmax, 'both')
            iplot += 1

def plot_t_3D(srcxyz, vs, vp_vs_ratio, true, pred):
    vs = vs / 1000
    vp = vs * vp_vs_ratio
    misfit = pred - true
    vmax = np.percentile(true, 99)
    
    fig = plt.figure(figsize=(36, 17), facecolor='white')
    fig.tight_layout()
    fig.facecolor = 'white'
    plt.rcParams['axes.facecolor'] = 'w'

    ax0 = fig.add_subplot(494, projection='3d', facecolor='white')
    ax0.plot([srcxyz[0]/1000, 5], [srcxyz[1]/1000, srcxyz[1]/1000],[srcxyz[2]/1000, srcxyz[2]/1000], color='gray', linestyle='--')
    ax0.plot([srcxyz[0]/1000, srcxyz[0]/1000], [5, srcxyz[1]/1000], [srcxyz[2]/1000, srcxyz[2]/1000], color='gray', linestyle='--')
    ax0.plot([srcxyz[0]/1000, srcxyz[0]/1000], [srcxyz[1]/1000, srcxyz[1]/1000], [5, srcxyz[2]/1000], color='gray', linestyle='--')
    ax0.plot(srcxyz[0]/1000, srcxyz[1]/1000, srcxyz[2]/1000, 'r*', markersize=20)
    ax0.set_xlabel("X (km)", fontsize=12)
    ax0.set_ylabel("Y (km)", fontsize=12)
    ax0.tick_params(labelsize=12)
    ax0.set_title("$\mathbf{Source}$", fontsize=15)
    ax0.set_xlim(0, 5)
    ax0.set_ylim(0, 5)
    ax0.set_zlim(0, 5)
    ax0.set_box_aspect([1, 1, 1])
    ax0.invert_zaxis()
    ax0.view_init(azim=225)
    ax0.text2D(0, 0.9, "Z (km)", transform=ax0.transAxes,
        ha='left', va='top', fontsize=12)
    
    plot_3D(fig, xx, yy, zz, vs, 'Spectral', (4, 9, 5), "$\mathbf{V_S (km/s)}$", np.percentile(vs, 1), np.percentile(vs, 99), 'both', if_clb=True)
    scatter_3D(fig, xx, yy, zz, vp, 'Spectral', (4, 9, 6), "$\mathbf{V_P (km/s)}$", np.percentile(vp, 1), np.percentile(vp, 99), 3, True)
    
    iplot = 10
    for step in [14, 20, 30]:
        datagroup = [true[:, :, :, step, 0], pred[:, :, :, step, 0], misfit[:, :, :, step, 0], 
                     true[:, :, :, step, 1], pred[:, :, :, step, 1], misfit[:, :, :, step, 1], 
                     true[:, :, :, step, 2], pred[:, :, :, step, 2], misfit[:, :, :, step, 2]]
        
        namegroup = ["True $\mathregular{U_x}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "Predicted $\mathregular{U_x}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "Misfit $\mathregular{U_x}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "True $\mathregular{U_y}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "Predicted $\mathregular{U_y}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "Misfit $\mathregular{U_y}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "True $\mathregular{U_z}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "Predicted $\mathregular{U_z}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt),
                     "Misfit $\mathregular{U_z}$ " + "at $\mathbf{{{:.4g}}}$ $\mathbf{{s}}$".format(step*dt)]
        
        for data, name in zip(datagroup, namegroup):
            plot_3D(fig, xx, yy, zz, data, 'RdBu_r', (4, 9, iplot), name, -vmax, vmax, 'both')
            iplot += 1
