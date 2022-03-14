# This is a sample Python script.
# eos include various basic functionals
import numpy as np
import pickle
from pathlib import Path
import json
import itertools
from utils import tecplot_WriteRectilinearMesh

cupyReady = False
try:
    import cupy as cp

    cupyReady = True
except ImportError:
    cp = np

from grid import *


# implement ideal gas eos
class INCOMP_Base:
    rho = 1  # kg/m^3
    mu = 1e-6  # kg/(m.s)

    def __init__(self, *args, **kwargs):
        input_meta = kwargs.get("input_json", None)
        if input_meta:
            self.rho = input_meta["rho"]
            self.mu = input_meta["mu"]
            print("Load input meta")

    def nu(self):
        return self.mu / self.rho

    def showInfo(self):
        print("Density: " + str(self.rho))
        print("Kinematic viscosity: " + str(self.mu))


class IncompressibleSolver:
    grid = UniformGrid()
    eos = INCOMP_Base()

    t, tf, dt = 0, 2.0, 0.0
    nti, dti = 0, 0.01
    iter = 0
    tif = t

    cfl = 0.2
    U, UH = [], []
    K, Ksq = [], []
    filter = 1

    output_dir = []
    b_dumpfile = False
    b_Resume = False

    b_Cupy = True
    cupy_module = cp

    verbose = 0

    def __init__(self, *args, **kwargs):
        input_meta = kwargs.get("input_json", None)
        if input_meta:
            self.t = input_meta["T start"]
            self.tf = input_meta["T end"]
            self.cfl = input_meta["CFL"]
            self.nti = input_meta["Frame index"]
            self.dti = input_meta["Frame interval"]
            self.iter = input_meta["Total iteration"]
            self.b_Resume = input_meta["Resume"]
            self.b_dumpfile = input_meta["Write output"]
            self.verbose = input_meta["Verbose"]
            self.b_Cupy = input_meta["GpuFlag"] and cupyReady
            print("Solver::Load input meta")

    def set_grid(self, grid):
        self.grid = grid

    def set_eos(self, eos):
        self.eos = eos

    def set_params(self):
        self.gen_buffer()
        self.init_buffer()

    def rfft(self, u):
        np = self.cupy_module
        return np.fft.fftn(u)

    def irfft(self, u):
        np = self.cupy_module
        return np.real(np.fft.ifftn(u))

    def fftfreq(self, i):
        np = self.cupy_module
        return np.fft.fftfreq(self.rfft_size[i], d=self.grid.ds[i] / 2 / np.pi)

    def boundary_correction(self):
        def all_periodic():
            nb = self.grid.nblayer
            for k in range(self.grid.ndims):
                s0, s1, s2, s3 = [slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim, [
                    slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim
                s0[k + 1], s1[k + 1], s2[k + 1], s3[k + 1] = slice(0, nb), slice(nb, 2 * nb), \
                                                             slice(-2 * nb, -nb), slice(-nb, None)
                s0, s1, s2, s3 = tuple(s0), tuple(s1), tuple(s2), tuple(s3)
                self.U[s0] = self.U[s2]
                self.U[s3] = self.U[s1]

        all_periodic()
        if self.verbose > 2:
            print("Boundary Corrected")

    def to_fourier_vars(self, U, UH):
        if self.verbose > 3:
            print(" - Convert to fourier space")
        u = self.splitvars(U)
        uh = self.splitvars(UH)
        for i in range(u.shape[0]):
            uh[i] = self.rfft(u[i])

    def to_physical_vars(self, U, UH):
        if self.verbose > 3:
            print(" - Convert to physical space")
        u = self.splitvars(U)
        uh = self.splitvars(UH)
        for i in range(u.shape[0]):
            u[i] = self.irfft(uh[i])

    def gen_buffer(self):
        np = self.cupy_module
        # self.rfft_size = tuple([i // 2 + 1 for i in self.grid.size])
        self.rfft_size = tuple([i - 2 * self.grid.nblayer for i in self.grid.size])

        if self.grid.b_Distribute:
            self.U = np.memmap(self.output_dir.joinpath('_cu'), dtype=np.float64, mode='w+',
                               shape=(self.grid.ndims, *self.grid.size))
            self.U = np.memmap(self.output_dir.joinpath('_cu'), dtype=np.float64, mode='r+',
                               shape=(self.grid.ndims, *self.grid.size))
            self.UH = np.memmap(self.output_dir.joinpath('_cuh'), dtype=np.complex128, mode='w+',
                                shape=(self.grid.ndims, *self.rfft_size))
            self.UH = np.memmap(self.output_dir.joinpath('_cuh'), dtype=np.complex128, mode='r+',
                                shape=(self.grid.ndims, *self.rfft_size))
            self.K = np.memmap(self.output_dir.joinpath('_ck'), dtype=np.float64, mode='w+',
                               shape=(self.grid.ndims, *self.rfft_size))
            self.K = np.memmap(self.output_dir.joinpath('_ck'), dtype=np.float64, mode='r+',
                               shape=(self.grid.ndims, *self.rfft_size))
            self.Ksq = np.memmap(self.output_dir.joinpath('_cksq'), dtype=np.float64, mode='w+',
                                 shape=(1, *self.rfft_size))
            self.Ksq = np.memmap(self.output_dir.joinpath('_cksq'), dtype=np.float64, mode='r+',
                                 shape=(1, *self.rfft_size))
        else:
            self.U = np.zeros((self.grid.ndims, *self.grid.size))
            self.UH = np.zeros((self.grid.ndims, *self.rfft_size)) + 0j
            self.K = np.zeros((self.grid.ndims, *self.rfft_size))
            self.Ksq = np.zeros((1, *self.rfft_size))
        self.U[:], self.UH[:] = 0, 0 + 0j
        self.K[:], self.Ksq[:] = 0, 0

        if self.verbose > 1:
            print("Allocated buffer for solver.")

    def refresh_buffer(self, mode='r+'):
        pass

    def compute_wavenumber_and_filter(self):
        # Generate wavenumber matrix
        # assume 1 domain with 0 ghost layer
        np = self.cupy_module
        kcoords = []
        for i in range(self.grid.ndims):
            kcoords = (*kcoords, self.fftfreq(i))

        if self.grid.b_Distribute:
            self.K = np.memmap(self.output_dir.joinpath('_mesh'), dtype=np.float64, mode='w+',
                               shape=(self.grid.ndims, *self.rfft_size))
            np.stack(np.meshgrid(*kcoords, indexing='ij', copy=False), out=self.K)
            self.K.flush()
            self.K = np.memmap(self.output_dir.joinpath('_mesh'), dtype=np.float64, mode='r',
                               shape=(self.grid.ndims, *self.rfft_size))
        else:
            self.K = np.stack(np.meshgrid(*kcoords, indexing='ij'))

        self.Ksq[0] = self.get_magsqr(self.K)

        # Generate circular filter
        thresh = 0.65
        max_k = np.max(self.K)
        circle = np.sqrt(self.Ksq)
        cphi = thresh * max_k
        filterfac = 23.6
        self.filter = np.exp(-filterfac * (circle - cphi) ** 4.)
        self.filter[circle <= cphi] = 1

    def init_buffer(self):
        np = self.cupy_module
        self.refresh_buffer()
        self.compute_wavenumber_and_filter()

        # Generate initial condition
        for ii in self.grid.domain:
            if self.verbose > 2:
                print("\t Fill initial condition in block %s" % (str(ii)))
            slc = (slice(None), *ii)
            u = self.splitvars(self.U[slc])
            uh = self.splitvars(self.UH)
            k, ksq = self.K, self.Ksq

            # Taylor green
            x, y = np.asarray(self.grid.mesh[slc][0]), np.asarray(self.grid.mesh[slc][1])
            u[0] = np.cos(np.pi * x) * np.sin(np.pi * y)
            u[1] = -np.sin(np.pi * x) * np.cos(np.pi * y)
            u[:] += 1e-4 * (np.random.rand(*u.shape) - 0.5)

            self.to_fourier_vars(u, uh)
            uh *= self.filter
            self.projection(uh, k, ksq)
            self.to_physical_vars(u, uh)

        self.grid.refresh_buffer()
        self.refresh_buffer()
        self.boundary_correction()

    def advection(self, A, u, uh, k):
        for p in range(self.grid.ndims):
            Ap = 0
            # Compute advection
            for q in range(self.grid.ndims):
                dudq = self.irfft(1j * k[q] * uh[p])
                Ap += dudq * u[q]
            A[p] = self.rfft(Ap)

    def projection(self, uh, k, ksq):
        # perform projection
        np = self.cupy_module
        ksq.flat[0] = 1
        uh -= np.sum(k * uh, axis=0, keepdims=True) * k / ksq
        ksq.flat[0] = 0

    def solve(self):
        E0 = np.sum(self.get_magsqr(self.U))
        while self.t < self.tf:
            if self.verbose > 1:
                print("\t [%d] T = %f Advance solution, E = %f" % (
                    self.iter, self.t, np.sum(self.get_magsqr(self.U)) / E0 * 100))

            self.compute_cfl()  # modified dt
            # assert self.dt >= 0, "dt is not positive real or zero!"
            if (self.t + 1.1 * self.dt) >= self.tif:
                self.dt = self.tif - self.t
            else:
                self.dt = (self.tif - self.t) / ((self.tif - self.t) // self.dt + 1)
            # modify dt to make it an integer multiplier of tif - t

            self.t += self.dt

            self.iter += 1

            self.update()

            self.boundary_correction()

            if self.t == self.tif:
                self.dump_solution(self.nti)
                self.nti += 1
                self.tif += self.dti

    def compute_cfl(self):
        np = self.cupy_module
        s = float('inf')
        self.refresh_buffer()
        for ii in self.grid.domain:
            if self.verbose > 3:
                print("\t T = %f Computing CFL condition for domain %s" % (self.t, str(ii)))
            slc = (slice(None), *ii)
            u = self.splitvars(self.U[slc])
            max_u = np.max(np.sqrt(np.sum(u ** 2, axis=0)))

            if max_u == 0:
                s = 1e10
            else:
                for p in range(self.grid.ndims):
                    s = np.minimum(s, self.grid.ds[p] / max_u)

        self.dt = self.cfl * s

    def update(self):
        if self.dt == 0:
            return
        np = self.cupy_module
        self.refresh_buffer()
        for ii in self.grid.domain:
            if self.verbose > 2:
                print("\t T = %f Solving mesh decomposition %s" % (self.t, str(ii)))
            qc = (slice(None), *ii)
            u = self.splitvars(self.U[qc])
            # uh = self.splitvars(self.UH[qc])
            # k, ksq = self.K[qc], self.Ksq[qc]
            uh = self.splitvars(self.UH)
            k, ksq = self.K, self.Ksq
            A = np.zeros_like(uh)

            # Step n -> n+{1/2}

            self.advection(A, u, uh, k)
            A *= self.filter
            self.projection(A, k, ksq)
            uh_star = (uh - 0.5 * self.dt * A) / (1.0 + 0.5 * self.dt * self.eos.nu() * ksq)
            self.to_physical_vars(u, uh_star)

            # Step n -> n+1
            # self.advection(A, u, uh_star, k)
            self.weno5flux(A, self.U)
            A *= self.filter
            self.projection(A, k, ksq)
            uh[:] = (uh - self.dt * A) / (1.0 + self.dt * self.eos.nu() * ksq)

            uh *= self.filter
            self.projection(uh, k, ksq)
            self.to_physical_vars(u, uh)
        self.refresh_buffer()

    def weno5flux(self, A, u):
        np = self.cupy_module
        # stencil op include all required stencil operations for all axes
        self.boundary_correction()
        A.fill(0 + 0j)
        stencil_op = self.grid.stencil_op_collection[0]

        for p in range(self.grid.ndims):  # xp
            a = np.max(np.abs(u[p]))  # characteristic speed
            a = np.array([a] * self.grid.ndims)
            a[p] *= 2
            a = a.reshape((self.grid.ndims, *([1] * self.grid.ndims)))

            frw = 0.5 * u[p:(p + 1)] * u + a * u
            flw = np.roll(0.5 * u[p:(p + 1)] * u - a * u, -1, axis=1 + p)

            _, qc, _, _, _, _, _, qp, qm, qpp, qmm = stencil_op[p]

            ## Right Flux
            # Choose the positive fluxes, 'v', to compute the left cell boundary flux:
            # $u_{i+1/2}^{-}$
            fr = frw[qc]
            frmm = frw[qmm]
            frm = frw[qm]
            frp = frw[qp]
            frpp = frw[qpp]

            # Polynomials
            p0 = (2 * frmm - 7 * frm + 11 * fr) / 6
            p1 = (-frm + 5 * fr + 2 * frp) / 6
            p2 = (2 * fr + 5 * frp - frpp) / 6

            # Smooth Indicators (Beta factors)
            w0 = 13 / 12 * (frmm - 2 * frm + fr) ** 2 + 1 / 4 * (frmm - 4 * frm + 3 * fr) ** 2
            w1 = 13 / 12 * (frm - 2 * fr + frp) ** 2 + 1 / 4 * (frm - frp) ** 2
            w2 = 13 / 12 * (fr - 2 * frp + frpp) ** 2 + 1 / 4 * (3 * fr - 4 * frp + frpp) ** 2

            # Constants
            d0 = 1 / 10
            d1 = 6 / 10
            d2 = 3 / 10
            epsilon = 1e-6

            # Alpha weights
            w0 = d0 / (epsilon + w0) ** 2
            w1 = d1 / (epsilon + w1) ** 2
            w2 = d2 / (epsilon + w2) ** 2
            alphasum = w0 + w1 + w2

            # ENO stencils weights
            w0 = w0 / alphasum
            w1 = w1 / alphasum
            w2 = w2 / alphasum

            # Numerical Flux at cell boundary, $u_{i+1/2}^{-}$;
            hn = w0 * p0 + w1 * p1 + w2 * p2

            ## Left Flux
            # Choose the negative fluxes, 'u', to compute the left cell boundary flux:
            # $u_{i-1/2}^{+}$
            fl = flw[qc]
            flmm = flw[qmm]
            flm = flw[qm]
            flp = flw[qp]
            flpp = flw[qpp]

            # Polynomials
            p0 = (-flmm + 5 * flm + 2 * fl) / 6
            p1 = (2 * flm + 5 * fl - flp) / 6
            p2 = (11 * fl - 7 * flp + 2 * flpp) / 6

            # Smooth Indicators (Beta factors)
            w0 = 13 / 12 * (flmm - 2 * flm + fl) ** 2 + 1 / 4 * (flmm - 4 * flm + 3 * fl) ** 2
            w1 = 13 / 12 * (flm - 2 * fl + flp) ** 2 + 1 / 4 * (flm - flp) ** 2
            w2 = 13 / 12 * (fl - 2 * flp + flpp) ** 2 + 1 / 4 * (3 * fl - 4 * flp + flpp) ** 2

            # Constants
            d0 = 3 / 10
            d1 = 6 / 10
            d2 = 1 / 10
            epsilon = 1e-6

            # Alpha weights
            w0 = d0 / (epsilon + w0) ** 2
            w1 = d1 / (epsilon + w1) ** 2
            w2 = d2 / (epsilon + w2) ** 2
            alphasum = w0 + w1 + w2

            # ENO stencils weights
            w0 = w0 / alphasum
            w1 = w1 / alphasum
            w2 = w2 / alphasum

            # Numerical Flux at cell boundary, $u_{i-1/2}^{+}$;
            hp = w0 * p0 + w1 * p1 + w2 * p2

            A += (hp - np.roll(hp, 1, axis=1 + p) + hn - np.roll(hn, 1, axis=1 + p)) / self.grid.ds[p]

        self.to_fourier_vars(A, A)

    def get_magsqr(self, vars):
        return np.sum(vars ** 2, axis=0)

    def splitvars(self, U):
        v1 = U
        return v1

    def dump_solution(self, n):
        if not self.b_dumpfile:
            return False
        self.refresh_buffer()
        rn = "result.%06d.tec" % (self.nti,)

        for ii in self.grid.domain:
            slc = (slice(None), *ii)
            if self.b_Cupy:
                u = self.splitvars(self.U[slc].get())
            else:
                u = self.splitvars(self.U[slc])
            uvars = [("u_%d" % i, u[i].flatten(order='F')) for i in range(self.grid.ndims)]
            vars = uvars
            self.grid.dumpToTecplot(self.output_dir.joinpath(rn), vars)
        return True

    def dump_meta(self):
        if not self.b_dumpfile:
            return False
        with open(self.output_dir.joinpath('meta.pkl'), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True

    def set_output_dir(self, indir):
        with Path(indir) as output_dir:
            assert not output_dir.is_file(), "Input directory is a file!"
            assert not output_dir.is_reserved(), "Input directory is reserved!"
            assert output_dir.is_dir(), "Input directory does not exist!"

            self.output_dir = output_dir
            self.b_dumpfile = True

            print("Solver::Current output directory is set to %s" % output_dir)


def clear_output_dir(dir):
    [f.unlink() for f in Path(dir).glob("*") if f.is_file()]
    return True


def load_input(input_meta='ns.json'):
    with open(input_meta, ) as handle:
        return json.load(handle)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Opening JSON file
    meta = load_input('ns.json')
    clear_output_dir(meta["output_dir"])

    eos_model = INCOMP_Base(input_json=meta["eos"])
    eos_model.showInfo()

    grid = UniformGrid()
    grid.nblayer = 2  # spectral grid
    grid.size = tuple([i + 2 * grid.nblayer for i in meta["grid"]["size"]])
    grid.rect = tuple(meta["grid"]["rect"])
    grid.n_decomposition = tuple(meta["grid"]["decompose"])
    grid.set_output_dir(meta["output_dir"])
    grid.b_Distribute = meta["grid"]["distributive"]
    grid.set_params()
    grid.create_stencil_op_collection()

    solver = IncompressibleSolver(input_json=meta["solver"])
    solver.set_output_dir(meta["output_dir"])
    solver.set_grid(grid)
    solver.set_eos(eos_model)
    solver.set_params()
    solver.solve()
    # solver.dump_meta()
