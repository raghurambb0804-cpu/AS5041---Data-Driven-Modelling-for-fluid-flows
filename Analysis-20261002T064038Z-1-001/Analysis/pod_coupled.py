import glob
import numpy as np
from read_flowfiles import read_flow_file
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from read_files import *
import torch
import time
import math
import matplotlib.tri as mtri
import imageio
from tqdm import trange
import os
import pyvista as pv

files = sorted(glob.glob("../flow.*.xyz"))
xgrid = X.reshape(-1)
ygrid = Y.reshape(-1)

all_Rh, all_U, all_V, all_W, all_P, all_T = [], [], [], [], [], []


def plot_pod_modes(modes, rank, filename):
    """
    Plot POD modes as a subplot grid and save to PNG.

    modes: 2D array (N spatial points × #modes)
    rank: number of modes to plot
    filename: name of PNG file to save
    """

    ncols = math.ceil(math.sqrt(rank))
    nrows = math.ceil(rank / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    
    # handle 1×1 case
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]
    axs = np.array(axs).ravel()

    for i in range(rank):
        ax = axs[i]
        mode = modes[:, i].detach().cpu()
        ct = ax.tricontourf(xgrid, ygrid, mode, levels=100)
        fig.colorbar(ct, ax=ax)
        ax.set_title(f"Mode {i+1}")
        ax.set_aspect('equal')

    # hide unused axes
    for j in range(rank, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    #plt.close(fig)

# --- Read snapshots ---
for count, f in enumerate(files, start=1):
    print(count)
    Rh, U, V, W, P, T, NI, NJ, NK, n = read_flow_file(f)
    all_Rh.append(Rh)
    all_U.append(U)
    all_V.append(V)
    all_W.append(W)
    all_P.append(P)
    all_T.append(T)

# Convert lists to snapshot matrices (snapshots x gridpoints)
all_Rh = np.array(all_Rh)[:, 0, :]
all_U  = np.array(all_U)[:, 0, :]
all_V  = np.array(all_V)[:, 0, :]
all_W  = np.array(all_W)[:, 0, :]
all_P  = np.array(all_P)[:, 0, :]
all_T  = np.array(all_T)[:, 0, :]

# Now N = number of spatial points
N = all_Rh.shape[1]
print("Grid points =", N)

# Transpose -> (gridpoints x snapshots)
all_Rh = all_Rh.T
all_U  = all_U.T
all_V  = all_V.T
all_W  = all_W.T
all_P  = all_P.T
all_T  = all_T.T

print(NI,NJ,NK)
# Stack variables vertically
all = np.vstack([all_Rh, all_U, all_V, all_P])
print(np.shape(all))
# --- Compute SVD on GPU ---
X = torch.tensor(all, dtype=torch.float64, device='cuda')
U, S, Vh = torch.linalg.svd(X, full_matrices=False)

#S_cpu = S.detach().cpu().numpy()
#energy = S_cpu**2
#energy_pct = 100.0 * energy / energy.sum()
#cum_energy_pct = np.cumsum(energy_pct)
#
#print("Top singular values (first 10):", S_cpu[:10])

# Reconstruction error check
rank = [1,2,4,8,16,32,64,128]
error = []
for r in rank:
    X_reconstruct = (U[:, :r] * S[:r]) @ Vh[:r, :]
    err = torch.norm(X - X_reconstruct) / torch.norm(X)
    error.append(err.item())
print(error)

plt.figure(figsize=(8,4))

plt.plot(rank, error, marker='o', linewidth=2, markersize=8)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Rank Truncation (r)", fontsize=14)
plt.ylabel("Relative Reconstruction Error", fontsize=14)
plt.title("Rank vs Reconstruction Error", fontsize=16)

plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig("pod_error_M0.5.png", dpi=300)
plt.show()



all_P_mean = np.mean(all_P, axis=1, keepdims=True)
all_P_fluc = all_P - all_P_mean

r = 1
X_reconstruct = (U[:, :r] * S[:r]) @ Vh[:r, :]


#Y = torch.tensor(all_P_fluc, dtype=torch.float64, device='cuda')
#U_fluc, S_fluc, Vh_fluc = torch.linalg.svd(Y, full_matrices=False)

# density
#plot_pod_modes(U[:N, :], 1, "rho_modes.png")

# u-velocity
#plot_pod_modes(U[N:2*N, :], 4, "u_modes.png")
#
## v-velocity
#plot_pod_modes(U[2*N:3*N, :], 4, "v_modes.png")
#
## pressure
#plot_pod_modes(U[3*N:4*N, :], 4, "p_modes.png")
#
## pressure fluctuation
#plot_pod_modes(U_fluc[:N, :], 4, "p_fluc_modes.png")


#def export_pod_surface(X_reconstruct, xgrid, ygrid, prefix="pod_", folder="pod"):
#    os.makedirs(folder, exist_ok=True)
#
#    Xr = X_reconstruct.detach().cpu().numpy()
#    nt = Xr.shape[1]
#    Npts = xgrid.size
#
#    pts = np.column_stack([xgrid.ravel(), ygrid.ravel(), np.zeros(Npts)])
#
#    for t in range(nt):
#        snap = Xr[:, t]
#
#        rho = snap[0*Npts:1*Npts]
#        u   = snap[1*Npts:2*Npts]
#        v   = snap[2*Npts:3*Npts]
#        p   = snap[3*Npts:4*Npts]
#
#        pd = pv.PolyData(pts)
#
#        pd["rho"] = rho
#        pd["u"]   = u
#        pd["v"]   = v
#        pd["p"]   = p
#
#        # Triangulate (same as tricontourf logic!)
#        surf = pd.delaunay_2d()
#
#        fname = os.path.join(folder, f"{prefix}{t:04d}.vtp")
#        surf.save(fname)
#        print("✓ wrote surface", fname)


#export_pod_surface(X_reconstruct, xgrid, ygrid, prefix="pod_",folder="pod")

def export_vtk_flat(folder, prefix, NI, NJ, x_flat, y_flat,
                    rho_flat, u_flat, v_flat, p_flat):

    os.makedirs(folder, exist_ok=True)
    Npts = NI * NJ

    if rho_flat.ndim == 1:
        rho_flat = rho_flat[:,None]
        u_flat   = u_flat[:,None]
        v_flat   = v_flat[:,None]
        p_flat   = p_flat[:,None]

    nt = rho_flat.shape[1]
    Z = np.zeros_like(x_flat)

    # reshape grid once
    X2 = x_flat.reshape(NI,NJ, order='F')
    Y2 = y_flat.reshape(NI,NJ, order='F')
    Z2 = Z.reshape(NI,NJ, order='F')

    for t in range(nt):
        grid = pv.StructuredGrid(X2, Y2, Z2)

        grid["rho"] = rho_flat[:,t]
        grid["u"]   = u_flat[:,t]
        grid["v"]   = v_flat[:,t]
        grid["p"]   = p_flat[:,t]

        fname = os.path.join(folder, f"{prefix}{t:04d}.vtk")
        grid.save(fname)
        print("✅ wrote", fname)

X_reconstruct_cpu = X_reconstruct.detach().cpu()
all_Rh_mode = X_reconstruct_cpu[:N,:]
all_U_mode  = X_reconstruct_cpu[N:2*N,:]
all_V_mode  = X_reconstruct_cpu[2*N:3*N,:]
all_P_mode  = X_reconstruct_cpu[3*N:4*N,:]
export_vtk_flat("pod_vtk", "pod_", NI[0], NJ[0], xgrid, ygrid,
                all_Rh_mode, all_U_mode, all_V_mode, all_P_mode)
