import numpy as np
import pickle
from pathlib import Path
import json
import itertools
from utils import tecplot_WriteRectilinearMesh

try:
    import cupy as cp
except ImportError:
    cp = np
    pass


class UniformGrid:
    size = (16, 16, 16)
    rect = ((-1, 0), (0, 1), (1, 2))
    ds = []
    nelem = 0
    # nvars = 0
    ndims = 0

    coords = []  # coordinates
    mesh = []  # grid

    nblayer = 2
    n_decomposition = (1, 1, 1)  # (2,2,2) etc
    domain = []
    stencil_op_collection = []

    output_dir = []
    b_Distribute = False

    def set_output_dir(self, indir):
        with Path(indir) as output_dir:
            assert not output_dir.is_file(), "Input directory is a file!"
            assert not output_dir.is_reserved(), "Input directory is reserved!"
            assert output_dir.is_dir(), "Input directory does not exist!"

            self.output_dir = output_dir
            self.b_Distribute = True

            print("Grid::Current output directory is set to %s" % output_dir)

    def set_params(self):
        self.nelem = np.prod(np.asarray(self.size))
        self.ndims = len(self.size)

        self.gen_buffer()
        self.decompose_domain()

    def gen_buffer(self):
        self.coords = []
        self.ds = []
        for i in range(len(self.size)):
            n = self.size[i]
            self.coords = (*self.coords, self.rect[i][0] +
                           (self.rect[i][1] - self.rect[i][0]) * (np.arange(-self.nblayer, n - self.nblayer) + 0.5) / (
                                   n - 2 * self.nblayer))
            self.ds = (*self.ds, self.coords[-1][1] - self.coords[-1][0])

        if self.b_Distribute:
            self.mesh = np.memmap(self.output_dir.joinpath('_mesh'), dtype=np.float64, mode='w+',
                                  shape=(self.ndims, *self.size))
            np.stack(np.meshgrid(*self.coords, indexing='ij', copy=False), out=self.mesh)
            self.mesh.flush()
            self.mesh = np.memmap(self.output_dir.joinpath('_mesh'), dtype=np.float64, mode='r',
                                  shape=(self.ndims, *self.size))
        else:
            self.mesh = np.stack(np.meshgrid(*self.coords, indexing='ij'))

    def refresh_buffer(self):
        if self.b_Distribute:
            self.mesh = []
            self.mesh = np.memmap(self.output_dir.joinpath('_mesh'), dtype=np.float64, mode='r',
                                  shape=(self.ndims, *self.size))

    def decompose_domain(self):
        def get_chunk(i, nchunks, ndim):
            start = i * (ndim - 2 * self.nblayer) // nchunks + self.nblayer
            end = (i + 1) * (ndim - 2 * self.nblayer) // nchunks + self.nblayer
            if i == (nchunks - 1):
                start = i * (ndim - 2 * self.nblayer) // nchunks + self.nblayer
                end = -self.nblayer

            if i == 0:
                start = self.nblayer
                end = (i + 1) * (ndim - 2 * self.nblayer) // nchunks + self.nblayer

            if start == 0:
                start = None
            if end == 0:
                end = None
            return slice(start, end)

        self.domain = []
        for k in itertools.product(*tuple(map(range, self.n_decomposition))):
            slc = list(map(get_chunk, k, self.n_decomposition, self.size))
            self.domain.append(slc)

    def get_stencil_shifted(self, slc=(), axis=-1, shift=0):
        new_slc = slc.copy()
        assert np.abs(shift) <= self.nblayer, "stencil shift > num of boundary layer is not permitted"
        new_start, new_stop = slc[axis].start - shift, slc[axis].stop - shift
        new_stop = new_stop if new_stop != 0 else None
        new_slc[axis] = slice(new_start, new_stop)
        return new_slc

    def get_stencil_extended(self, slc=(), axis=-1, padding=(0, 0)):
        new_slc = slc.copy()
        assert np.abs(padding[0]) <= self.nblayer, "stencil padding left  > num of boundary layer is not permitted"
        assert np.abs(padding[1]) <= self.nblayer, "stencil padding right > num of boundary layer is not permitted"
        new_start, new_stop = slc[axis].start - padding[0], slc[axis].stop + padding[1]
        new_stop = new_stop if new_stop != 0 else None
        new_slc[axis] = slice(new_start, new_stop)
        return new_slc

    def get_stencil_operators(self, q, k, ndims):
        # Get the stencil without ghost layer
        qc = [slice(None), *q]

        # Get the stencil with ghost layer
        qe = [*q].copy()
        for i in range(ndims):
            qe = self.get_stencil_extended(qe, axis=i, padding=(2, 2))  # n + 2
        qe = [slice(None), *qe]

        # Get the extended stencil with shifting
        dq = [slice(None), *[slice(2, -2)] * self.ndims]
        qce, qpe, qme = dq.copy(), dq.copy(), dq.copy()
        qce[1 + k], qpe[1 + k], qme[1 + k] = slice(1, -1), slice(2, None), slice(-2)

        # Get the regular stencil with shifting
        qp, qm, qpp, qmm = dq.copy(), dq.copy(), dq.copy(), dq.copy()
        qp[1 + k], qm[1 + k] = slice(3, -1), slice(1, -3)
        qpp[1 + k], qmm[1 + k] = slice(4, None), slice(None, -4)

        # Get the biased extended stencil
        dqp, dqm = [slice(None)] * (1 + self.ndims), [slice(None)] * (1 + self.ndims)
        dqp[k + 1], dqm[k + 1] = slice(1, None), slice(-1)  # n + 1

        return tuple(qe), tuple(qc), tuple(qce), tuple(qpe), tuple(qme), tuple(dqp), tuple(dqm), \
            tuple(qp), tuple(qm), tuple(qpp), tuple(qmm)

    def create_stencil_op_collection(self):
        self.stencil_op_collection = [[[] for _ in range(self.ndims)] for _ in range(np.product(self.n_decomposition))]
        for p in range(self.ndims):
            for q in range(np.product(self.n_decomposition)):
                self.stencil_op_collection[q][p] = self.get_stencil_operators(self.domain[q], p, self.ndims)

    def dumpgridToTecplot(self):
        # Coordinates
        X = self.coords[0]
        Y = self.coords[1]
        if self.ndims == 2:
            Z = [0]
        else:
            Z = self.coords[2]

        # Data
        nodeid = []
        id = 0
        for k in range(len(Z)):
            for j in range(len(Y)):
                for i in range(len(X)):
                    nodeid = nodeid + [id]
                    id = id + 1

        # Write the data to Tecplot format
        if self.ndims == 2:
            vars = (("nodeid", nodeid), ("point_0", self.mesh[0].flatten(order='F')),
                    ("point_1", self.mesh[1].flatten(order='F')))
            tecplot_WriteRectilinearMesh(self.output_dir.joinpath("grid.tec"), X, Y, [], vars)  # 2D
        else:
            vars = (("nodeid", nodeid), ("point_0", self.mesh[0].flatten(order='F')),
                    ("point_1", self.mesh[1].flatten(order='F')), ("point_2", self.mesh[2].flatten(order='F')))
            tecplot_WriteRectilinearMesh(self.output_dir.joinpath("grid.tec"), X, Y, Z, vars)  # 3D

    def dumpToTecplot(self, filename, vars):
        # Coordinates
        if self.nblayer > 0:
            slc = slice(self.nblayer, -self.nblayer)
        else:
            slc = slice(None)
        X = self.coords[0][slc]
        Y = self.coords[1][slc]
        if self.ndims == 2:
            Z = [0]
        else:
            Z = self.coords[2][slc]

        # Write the data to Tecplot format
        # todo: for mesh higher than 3d get a selected 2d/3d slice to dump
        if self.ndims == 2:
            tecplot_WriteRectilinearMesh(filename, X, Y, [], vars)  # 2D
        else:
            tecplot_WriteRectilinearMesh(filename, X, Y, Z, vars)  # 3D


def load_input(input_meta='ns.json'):
    with open(input_meta, ) as handle:
        return json.load(handle)


if __name__ == '__main__':
    # Opening JSON file
    meta = load_input('ns.json')

    grid = UniformGrid()
    grid.set_output_dir(meta["output_dir"])
    grid.b_Distribute = meta["grid"]["distributive"]
    grid.set_params()
    grid.dumpgridToTecplot()
