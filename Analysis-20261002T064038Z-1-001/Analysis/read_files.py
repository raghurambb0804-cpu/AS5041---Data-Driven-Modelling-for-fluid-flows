import numpy as np
import matplotlib.pyplot as plt
def read_record(f, dtype):
    # read leading length
    nbytes = np.fromfile(f, dtype=np.int32, count=1)[0]
    count = nbytes // np.dtype(dtype).itemsize

    # read data
    arr = np.fromfile(f, dtype=dtype, count=count)

    # read trailing length
    _ = np.fromfile(f, dtype=np.int32, count=1)

    return arr

def read_fortran_grid(filename, dtype=np.float64):
    with open(filename, "rb") as f:

        # ---- record 1: number of blocks ----
        nblocks = read_record(f, np.int32)[0]
        print("Blocks:", nblocks)

        # ---- record 2: NI, NJ, NK for each block ----
        dims = read_record(f, np.int32).reshape(nblocks, 3)
        NI, NJ, NK = dims[:,0], dims[:,1], dims[:,2]

        Xgrids, Ygrids, Zgrids = [], [], []

        # ---- next records: grid blocks ----
        for b in range(nblocks):
            n = NI[b] * NJ[b] * NK[b]
        
            # read ONE record containing X,Y,Z
            raw = read_record(f, dtype)
        
            # sanity check
            if raw.size != 3 * n:
                raise ValueError(f"Block {b}: expected {3*n} values, got {raw.size}")
        
            X = raw[0:n    ].reshape(NI[b], NJ[b], NK[b])
            Y = raw[n:2*n  ].reshape(NI[b], NJ[b], NK[b])
            Z = raw[2*n:3*n].reshape(NI[b], NJ[b], NK[b])
        
            Xgrids.append(X)
            Ygrids.append(Y)
            Zgrids.append(Z)


    return Xgrids, Ygrids, Zgrids, NI, NJ, NK
X,Y,Z,NI,NJ,NK = read_fortran_grid("../grid_anim.xyz", dtype=np.float64)

X = np.array(X)
Y = np.array(Y)

X = X[0,:,:,0]
Y = Y[0,:,:,0]

print(np.shape(X))
#plt.figure()
#plt.plot(X[0,:,:,0], Y[0,:,:,0], 'k.', markersize=1)
#plt.axis('equal')
#plt.show()

