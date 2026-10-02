import numpy as np

def read_record(f, dtype):
    # leading record length
    reclen = np.fromfile(f, np.int32, 1)
    if len(reclen) == 0:
        raise EOFError
    nbytes = reclen[0]

    # data block
    data = np.fromfile(f, dtype, nbytes // np.dtype(dtype).itemsize)

    # trailing record length
    _ = np.fromfile(f, np.int32, 1)
    return data

def read_flow_file(filename, dtype=np.float64):

    with open(filename, "rb") as f:

        # number of blocks
        nblocks = read_record(f, np.int32)[0]

        # NI,NJ,NK,nanim for each block
        arr = read_record(f, np.int32)
        NI = arr[0::4]
        NJ = arr[1::4]
        NK = arr[2::4]
        nanim = arr[3::4]

        Rh = []
        U  = []
        V  = []
        W  = []
        P  = []
        T  = []

        for b in range(nblocks):
            ni = NI[b]
            nj = NJ[b]
            nk = NK[b]

            # read one giant record containing 6 fields
            data = read_record(f, dtype)

            total = ni * nj * nk

            # safety
            assert len(data) == 6 * total, \
                f"Mismatch: record has {len(data)}, expected {6*total}"

            # slice into 6 fields
            Rh.append(data[0*total:1*total])#.reshape(ni,nj,nk))
            U.append (data[1*total:2*total])#.reshape(ni,nj,nk))
            V.append (data[2*total:3*total])#.reshape(ni,nj,nk))
            W.append (data[3*total:4*total])#.reshape(ni,nj,nk))
            P.append (data[4*total:5*total])#.reshape(ni,nj,nk))
            T.append (data[5*total:6*total])#.reshape(ni,nj,nk))

    return Rh, U, V, W, P, T, NI, NJ, NK, nanim
