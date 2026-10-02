import glob
import numpy as np
from read_flowfiles import read_flow_file
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from read_files import *
import torch
import time
import os
#import matplotlib
#matplotlib.use("Agg")
import math
import pyvista as pv

files = sorted(glob.glob("../flow.*.xyz"))
xgrid = X.reshape(-1)
ygrid = Y.reshape(-1)

all_Rh, all_U, all_V, all_W, all_P, all_T = [], [], [], [], [], []
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

# Stack variables vertically
all = np.vstack([all_Rh, all_Rh*all_U, all_Rh*all_V, all_P])

dt = 0.0125
U_inf = 171.9665

# Snapshot matrices
all_LHS = all[:, :-1]   # X
all_RHS = all[:, 1:]    # Y

# Move to GPU
X = torch.tensor(all_LHS, dtype=torch.float64, device='cuda')
Y = torch.tensor(all_RHS, dtype=torch.float64, device='cuda')

full_dof, num_snap = X.shape

# Economy SVD of Y
U, S, Vh = torch.linalg.svd(Y, full_matrices=False)

r = 200
Ur = U[:, :r]   # POD basis (Y-based projection)

# Project X and Y onto reduced space
Xr = Ur.T @ X
Yr = Ur.T @ Y

# Form TLS concat
Z = torch.vstack((Xr, Yr))

# TLS SVD
U_t, S_t, Vh_t = torch.linalg.svd(Z, full_matrices=False)

# Partition U_t (cols)
# U_t = [U11  U12]
#       [U21  U22]

U11 = U_t[:r, :r]
U21 = U_t[r:, :r]

# TLS operator
A_tilde = U21 @ torch.linalg.inv(U11)

eigvals, eigvecs = torch.linalg.eig(A_tilde)

Ur_c = Ur.to(torch.complex128)

Phi = Ur_c @ eigvecs

x0 = X[:,0]

x0 = x0.to(torch.complex128)

b = torch.linalg.lstsq(Phi, x0).solution


X_rec = torch.empty((full_dof, num_snap), dtype=torch.complex128, device='cuda')
X_dmd = torch.empty((full_dof, num_snap),dtype=torch.complex128,device='cuda')
t = torch.arange(num_snap, device='cuda') * dt
omega = torch.log(eigvals) / dt
#angle = torch.angle(eigvals)
freq2 = omega.imag / (2 * torch.pi * dt)   # Hz
St = torch.abs(freq2) * 1.0 / U_inf
gamma = omega.real
#print(St)

# 2. Cast once
Phi   = Phi.to(torch.complex128)
omega = omega.to(torch.complex128)
b     = b.to(torch.complex128)
t     = t.to(torch.complex128)

# 3. Column-wise reconstruction
for k in range(num_snap):
    time_k = b * torch.exp(omega * t[k])     # (r,)
    X_dmd[:, k] = Phi @ time_k               # (full_dof,)
    #del time_k

# 4. Real field
X_dmd = X_dmd.real

# 6. Error per timestep (cheap)
error_r200 = torch.empty(num_snap, device='cuda')
for k in range(num_snap):
    error_r200[k] = torch.norm(X[:,k] - X_dmd[:,k]) / torch.norm(X[:,k])
#print(error.cpu().numpy())


#plt.figure(figsize=(5,4))
#plt.plot(torch.arange(num_snap).cpu().numpy() * dt, error_cpu, '-o')
#plt.xlabel("Time (s)")
#plt.ylabel("Relative Error")
#plt.title("TLS-DMD Reconstruction Error")
#plt.grid(True)
#plt.savefig("error vs time rank 100")
#plt.show()
#
#gamma_cpu = gamma.detach().cpu().numpy()
#St_cpu = St.detach().cpu().numpy()
#
#plt.figure(figsize=(5,4))
#plt.scatter(St_cpu,gamma_cpu)
#plt.xlabel("St")
#plt.ylabel("Gamma")
#plt.title("Growth rate vs Stouhal number")
#plt.grid(True)
##plt.savefig("error vs time rank 100")
#plt.show()

################ greedy ###############
# ==========================================================
# Greedy Mode Selection (Ohmichi et al., 2018 – Appendix B)
# ==========================================================

def greedy_mode_selection(Psi, Phi, Vand, K):
    """
    Psi  : reduced snapshots (r, N)
    Phi  : reduced eigenvectors (r, m)
    Vand : Vandermonde (m, N)
    K    : number of modes to select
    
    Returns:
        selected  : mode index list
        Psi_hat   : reconstructed reduced snapshots (r, N)
    """

    device = Psi.device
    Psi = Psi.to(torch.complex128)
    Phi = Phi.to(torch.complex128)
    Vand = Vand.to(torch.complex128)

    r, N = Psi.shape
    m = Phi.shape[1]

    R = Psi.clone()  # residual
    Psi_hat = torch.zeros_like(Psi)

    phi_norm2 = torch.sum(torch.conj(Phi) * Phi, dim=0)
    vand_norm2 = torch.sum(torch.conj(Vand) * Vand, dim=1)
    contrib2 = phi_norm2 * vand_norm2  # ||C_i||^2

    selected = []
    remaining = torch.ones(m, dtype=torch.bool, device=device)

    normR2 = torch.sum(torch.conj(R) * R).real

    for k in range(min(K, m)):

        S = R @ torch.conj(Vand).T   # (r,m)
        inner = torch.sum(torch.conj(Phi) * S, dim=0)  # <R, C_i>

        # compute error_i = ||R - C_i||^2
        error = normR2 - 2.0 * inner.real + contrib2.real
        error = error.real  # critical!

        # mask selected modes
        error[~remaining] = 1e100

        idx = torch.argmin(error).item()
        selected.append(idx)
        remaining[idx] = False

        # mode contribution
        Ci = Phi[:,idx:idx+1] @ Vand[idx:idx+1,:]
        Psi_hat += Ci
        R -= Ci

        normR2 = torch.sum(torch.conj(R) * R).real
        if normR2 < 1e-20:
            break

    return selected, Psi_hat

# Reduced snapshots (you already compute this)
Psi_tilde = Xr.to(torch.complex128)           # (r,N)
Phi_tilde = eigvecs.to(torch.complex128)      # (r,m)
Vand = torch.exp(omega[:,None] * t[None,:])   # (m,N)

K = 10
selected, Psi_hat = greedy_mode_selection(Psi_tilde, Phi_tilde, Vand, K)
print("Selected indices:", selected)

selected_2 = selected#[selected[1],selected[2]]
print(omega[selected_2])
Phi_sel_red = Phi_tilde[:, selected_2]   # (r,K)
Vand_sel = Vand[selected_2, :]           # (K,N)

A = Phi_sel_red @ Vand_sel             # (r,N)
x0 = Psi_tilde[:,0]  # first reduced snapshot
alpha = torch.linalg.lstsq(Phi_sel_red, x0).solution  # ← K × 1

Phi_sel_red = Phi_sel_red.to(torch.complex128)
omega_sel   = omega[selected_2].to(torch.complex128)
alpha       = alpha.to(torch.complex128)
Ur_c        = Ur_c.to(torch.complex128)
t           = t.to(torch.complex128)

for k in range(num_snap):
    time_k = alpha * torch.exp(omega_sel * t[k])
    psi_k  = Phi_sel_red @ time_k
    X_rec[:, k] = Ur_c @ psi_k

Xc = X.cpu().numpy()       # keep CPU
X_rec_cpu = X_rec.real.detach().cpu().numpy()

error_greedy = torch.empty(num_snap, device='cuda')
for k in range(num_snap):
    error_greedy[k] = torch.norm(X[:,k] - X_rec[:,k]) / torch.norm(X_rec[:,k])


print(error_greedy)

omega_sel = torch.log(eigvals[selected]) / dt
#angle = torch.angle(eigvals)
freq2_sel = omega_sel.imag / (2 * torch.pi * dt)   # Hz
St_sel = torch.abs(freq2_sel) * 1.0 / U_inf
gamma_sel = omega_sel.real
print(St_sel)
print(omega_sel)

gamma_cpu = gamma.detach().cpu().numpy()
St_cpu = St.detach().cpu().numpy()

gamma_sel_cpu = gamma_sel.detach().cpu().numpy()
St_sel_cpu = St_sel.detach().cpu().numpy()

# Remove largest gamma value (outlier)
idx = gamma_cpu.argmax()
gamma_f = np.delete(gamma_cpu, idx)
St_f = np.delete(St_cpu, idx)

plt.scatter(St_f, gamma_f, s=12, marker='o', label='All modes')
plt.scatter(St_sel_cpu, gamma_sel_cpu, s=80, marker='x', label='Compressed Sensing modes')

plt.axhline(0, linestyle='--', linewidth=1)

plt.xlabel("Strouhal number (St)", fontsize=14)
plt.ylabel("Growth rate ($\\gamma$)", fontsize=14)
plt.title("Growth rate vs Strouhal number at M=0.5, 0° AOA", fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("growth_St_M0_5.png", dpi=500, bbox_inches='tight')
plt.show()



#plt.tricontourf(xgrid, ygrid, rho_sel[:, 0], levels=100)
#plt.title("Greedy-Selected DMD Reconstruction (rho, snapshot 0)")
#plt.xlabel("Spatial Index")
#plt.ylabel("rho value")
#plt.show()

# --- make folder if missing ---
#stride = 20
#os.makedirs("frames", exist_ok=True)
#
#for frame in range(0,num_snap,stride):
#    plt.figure(figsize=(5,4))
#    plt.tricontourf(xgrid, ygrid, X_i[:N, frame], levels=100)
#    plt.title(f"Single mode animation, step {frame}")
#    plt.gca().set_aspect("equal")
#    plt.colorbar()
#    
#    # Save as frames/frame_0000.png
#    plt.savefig(f"frames/frame_{frame:04d}.png", dpi=150)
#    plt.close()  

 #plt.show()   # <-- disable for speed
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
        grid["p_fluc"]   = p_flat[:,t]

        fname = os.path.join(folder, f"{prefix}{t:04d}.vtk")
        grid.save(fname)
        print("✅ wrote", fname)


X_i = X_rec.real.detach().cpu()

all_Rh_mode = X_i[:N,:]
all_U_mode  = X_i[N:2*N,:]/all_Rh_mode
all_V_mode  = X_i[2*N:3*N,:]/all_Rh_mode
all_P_mode  = X_i[3*N:4*N,:]
p_fluc = all_P_mode - all_P_mode.mean(axis=1, keepdims=True)


export_vtk_flat("dmd_1mode_vtk", "dmd_mode_", NI[0], NJ[0], xgrid, ygrid,all_Rh_mode, all_U_mode, all_V_mode, p_fluc)
## --- assumes eigvals already computed (complex) ---
#eigvals = eigvals.detach().cpu().numpy()
#
#dt = 0.0125  # <<--- set your real timestep here!
#
#growth_rates = np.log(np.abs(eigvals)) / dt
#
#plt.figure()
#plt.plot(np.real(growth_rates), 'o')
#plt.axhline(0, linestyle='--')
#plt.xlabel("Mode index")
#plt.ylabel("Growth rate (1/s)")
#plt.title("Temporal Growth/Decay Rates of DMD Modes")
#plt.show()


