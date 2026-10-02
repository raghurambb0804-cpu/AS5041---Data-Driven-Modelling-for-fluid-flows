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
from scipy.signal import welch
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

dt = 0.0125
rho = 1.2
U_inf = 171.9665
fs = 1.0/dt
all_P = all_P*(rho*U_inf**2)

p_fluc = all_P- all_P.mean(axis=1, keepdims=True)

P_trailingedge = p_fluc[100,:]
P_trailingup  = p_fluc[250,:]
P_trailingdown = p_fluc[675,:]



f, Pxx = welch(P_trailingedge, fs, window='hann', nperseg=2048)
fup, Pxx_up = welch(P_trailingup, fs, window='hann', nperseg=2048)
fdown, Pxx_down = welch(P_trailingdown, fs, window='hann', nperseg=2048)

St = f/(dt*U_inf)
St_up = fup/(dt*U_inf)
St_down = fdown/(dt*U_inf)

plt.semilogy(St, Pxx)
plt.semilogy(St_up, Pxx_up)

plt.xlabel('Strouhal number (St)', fontsize=16)
plt.ylabel('PSD [Pa²/Hz]', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("PSD vs Strouhal number at M=0.5", fontsize=18)
plt.legend(["TE Downstream", "TE Upstream"], fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig("Downstream_TE.png", dpi=500)
plt.show()
