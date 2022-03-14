# This is a sample Python script.
# eos include various basic functionals
import numpy as np
import pickle
from pathlib import Path
import json
import itertools
from utils import tecplot_WriteRectilinearMesh

try:
    import cupy as cp
    cupyReady = True
except ImportError:
    cp = np
    cupyReady = False
    pass

from grid import *


# implement ideal gas eos
class EOS_Base:
    Rbar = 1  # universal gas constant J/K/mol
    M = 1  # molecular mass kg/mol
    gamma = 1.4  # heat capacity ratio
    rhoinf = 1  # kg/mˆ3
    pinf = 1  # Pascal
    EOSname = 'ideal gas'

    def __init__(self, *args, **kwargs):
        input_meta = kwargs.get("input_json", None)
        if input_meta:
            self.EOSname = input_meta["model"]
            self.Rbar = input_meta["rbar"]
            self.M = input_meta["molemass"]
            self.gamma = input_meta["gamma"]
            self.rhoinf = input_meta["rhoinf"]
            self.pinf = input_meta["pinf"]
            print("Load input meta")

    def querySoundSpeed(self, rho, p):
        return np.sqrt(self.gamma * (self.Rbar / self.M) * self.queryTemp(rho, p))

    def queryTemp(self, rho, p):
        return p / rho / (self.Rbar / self.M)

    def queryInternalEnergy(self, rho, p):
        return p / (self.gamma - 1) / rho

    def showInfo(self):
        print("EOS model Name: " + self.EOSname)
        print("Ambient density: " + str(self.rhoinf))
        print("Ambient pressure: " + str(self.pinf))
        print("Ambient temperature: " + str(self.queryTemp(self.rhoinf, self.pinf)))
        print("Ambient speed of sound: " + str(self.querySoundSpeed(self.rhoinf, self.pinf)))


class EulerSolver:
    grid = UniformGrid()
    eos = EOS_Base()

    t, tf, dt = 0, 2.0, 0
    nti, dti = 0, 0.01
    iter = 0
    tif = t

    cfl = 0.2
    U, W = [], []

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

    def gen_buffer(self):
        if self.grid.b_Distribute:
            self.U = np.memmap(self.output_dir.joinpath('_cu'), dtype=np.float64, mode='w+',
                               shape=(2 + self.grid.ndims, *self.grid.size))
            self.W = np.memmap(self.output_dir.joinpath('_cw'), dtype=np.float64, mode='w+',
                               shape=(2 + self.grid.ndims, *self.grid.size))
            self.U = np.memmap(self.output_dir.joinpath('_cu'), dtype=np.float64, mode='r+',
                               shape=(2 + self.grid.ndims, *self.grid.size))
            self.W = np.memmap(self.output_dir.joinpath('_cw'), dtype=np.float64, mode='r+',
                               shape=(2 + self.grid.ndims, *self.grid.size))
        else:
            self.U = np.zeros((2 + self.grid.ndims, *self.grid.size))
            self.W = np.zeros((2 + self.grid.ndims, *self.grid.size))
        self.U[:], self.W[:] = 0, 0

        if self.verbose > 1:
            print("Allocated buffer for solver.")

    def refresh_buffer(self, mode='r+'):
        if self.grid.b_Distribute:
            self.U, self.W = [], []
            self.U, self.W = np.memmap(self.output_dir.joinpath('_cu'), dtype=np.float64, mode='r+',
                                       shape=(2 + self.grid.ndims, *self.grid.size)), \
                             np.memmap(self.output_dir.joinpath('_cw'), dtype=np.float64, mode='r+',
                                       shape=(2 + self.grid.ndims, *self.grid.size))
            if self.verbose > 3:
                print("Reconstruct memory mapping.")

    def init_buffer(self):
        rhol, ul, pl = 12.5 * self.eos.rhoinf, np.zeros(
            (self.grid.ndims, *np.ones(self.grid.ndims, dtype=int).tolist())), 40.0 * self.eos.pinf
        rhor, ur, pr = 1.0 * self.eos.rhoinf, np.zeros(
            (self.grid.ndims, *np.ones(self.grid.ndims, dtype=int).tolist())), 1.0 * self.eos.pinf

        center = np.zeros((self.grid.ndims, *np.ones(self.grid.ndims, dtype=int).tolist()))
        for i in range(self.grid.ndims):
            c = (-0.2, -0.25, 0.35 / 2, 0.1)
            center[-1 - i] = c[-i] * np.diff(self.grid.rect[i])[0]

        radius = 0.2 * np.diff(self.grid.rect[:]).reshape(self.grid.ndims,
                                                          *np.ones(self.grid.ndims, dtype=int).tolist()) \
                 + np.zeros((self.grid.ndims, *np.ones(self.grid.ndims, dtype=int).tolist()))

        self.refresh_buffer()
        for q in self.grid.domain:
            if self.verbose > 2:
                print("\t Fill initial condition in block %s" % (str(q)))
            slc = (slice(None), *q)

            outer = np.sum((self.grid.mesh[slc] - center) ** 2 / (radius ** 2), axis=0)
            outer = np.logical_or((outer > 1), np.logical_not(np.isfinite(outer)))

            rho, u, p = self.splitvars(self.W[slc])
            self.W[slc] = 0
            rho += outer * rhor + np.logical_not(outer) * rhol
            u += outer * ur + np.logical_not(outer) * ul
            p += outer * pr + np.logical_not(outer) * pl

        self.grid.refresh_buffer()
        self.refresh_buffer()
        for q in range(np.product(self.grid.n_decomposition)):
            if self.verbose > 2:
                print("\t T = %f Init::To conserved vars decomposition %s" % (self.t, str(q)))
            qe, qc = self.grid.stencil_op_collection[q][0][:2]
            self.to_conserved_vars(self.U[qc], self.W[qc])
        # self.to_conserved_vars(self.U, self.W)
        self.refresh_buffer()

        self.boundary_correction()

    def solve(self):
        while self.t < self.tf:
            for q in range(np.product(self.grid.n_decomposition)):
                if self.verbose > 2:
                    print("\t T = %f Compute CFL::To primitive vars decomposition %s" % (self.t, str(q)))
                qe, qc = self.grid.stencil_op_collection[q][0][:2]
                self.to_primitive_vars(self.U[qe], self.W[qe])

            self.compute_cfl()
            # assert self.dt >= 0, "dt is not positive real or zero!"
            if (self.t + 1.1 * self.dt) >= self.tif:
                self.dt = self.tif - self.t
            # assert self.dt >= 0, "dt is not positive real or zero!"
            self.t += self.dt

            self.iter += 1

            self.update()

            self.boundary_correction()

            if self.t == self.tif:
                self.dump_solution(self.nti)
                self.nti += 1
                self.tif += self.dti

    def compute_cfl(self):
        # self.to_primitive_vars(self.U, self.W)

        s = float('inf')
        self.refresh_buffer()
        for q in range(np.product(self.grid.n_decomposition)):
            if self.verbose > 3:
                print("\t T = %f Computing CFL condition for domain %s" % (self.t, str(q)))

            qe, qc = self.grid.stencil_op_collection[q][0][:2]

            if self.b_Cupy:
                Wi = self.cupy_module.asarray(self.W[qc])  # get slice to domain
            else:
                Wi = self.W[qc]  # get slice to domain

            rho, u, p = self.splitvars(Wi)
            assert (np.min(rho) > 0) & (np.min(p) >= 0), "CFL failed : invalid value for rho or p"
            a = self.eos.querySoundSpeed(rho, p)

            for k in range(self.grid.ndims):
                s = np.minimum(s, np.min(self.grid.ds[k] / (np.abs(u[k]) + a)))

        if self.b_Cupy:
            s = s.get()

        self.dt = 0.5 * self.cfl * s

        # print(self.dt)

    def boundary_correction(self):
        def all_periodic():
            nb = self.grid.nblayer
            for k in range(self.grid.ndims):
                s0, s1, s2, s3 = [slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim, [
                    slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim
                s0[-k - 1], s1[-k - 1], s2[-k - 1], s3[-k - 1] = slice(0, nb), slice(2 * nb - 1, nb - 1, -1), slice(
                    -nb - 1, -2 * nb - 1, -1), slice(-nb, None)
                s0, s1, s2, s3 = tuple(s0), tuple(s1), tuple(s2), tuple(s3)
                self.U[s0], self.U[s3] = self.U[s2], self.U[s1]

        def all_noslip():
            nb = self.grid.nblayer
            for k in range(self.grid.ndims):
                s0, s1, s2, s3 = [slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim, [
                    slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim
                s0[-k - 1], s1[-k - 1], s2[-k - 1], s3[-k - 1] = slice(0, nb), slice(2 * nb - 1, nb - 1, -1), slice(
                    -nb - 1, -2 * nb - 1, -1), slice(-nb, None)
                s0, s1, s2, s3 = tuple(s0), tuple(s1), tuple(s2), tuple(s3)

                self.U[s0], self.U[s3] = self.U[s1], self.U[s2]

                s0, s1, s2, s3 = [slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim, [
                    slice(None)] * self.U.ndim, [slice(None)] * self.U.ndim
                s0[-k - 1], s1[-k - 1], s2[-k - 1], s3[-k - 1] = slice(0, nb), slice(2 * nb - 1, nb - 1, -1), slice(
                    -nb - 1, -2 * nb - 1, -1), slice(-nb, None)
                s0[0] = s1[0] = s2[0] = s3[0] = slice(1, -1)
                s0, s1, s2, s3 = tuple(s0), tuple(s1), tuple(s2), tuple(s3)

                self.U[s0], self.U[s3] = -self.U[s1], -self.U[s2]

        all_noslip()
        if self.verbose > 1:
            print("Boundary Corrected")

    def update(self, b_cupy=True, cupy_module=np):
        def mc(a, b, np=np):
            if self.b_Cupy:
                np = self.cupy_module
            d = np.asarray(np.logical_and(a > 0, b > 0), dtype=float) - np.asarray(np.logical_and(a < 0, b < 0),
                                                                                   dtype=float)
            return d * np.minimum(np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)), np.abs(a + b) / 2)

            # self.to_primitive_vars(self.U, self.W)

        self.refresh_buffer()
        for q in range(np.product(self.grid.n_decomposition)):
            if self.verbose > 2:
                print("\t T = %f Solving mesh decomposition %s" % (self.t, str(q)))

            qe, qc = self.grid.stencil_op_collection[q][0][:2]

            dF = 0
            if self.b_Cupy:
                Wi = self.cupy_module.asarray(self.W[qe])  # get slice to domain
            else:
                Wi = self.W[qe]  # get slice to domain

            for k in range(self.grid.ndims):
                if self.verbose > 3:
                    print("T = %f Solving mesh dimension %d" % (self.t, k))

                _, _, qce, qpe, qme, dqp, dqm = self.grid.stencil_op_collection[q][k]

                Wc, Wp, Wm = Wi[qce], Wi[qpe], Wi[qme]

                dW = mc(Wp - Wc, Wc - Wm)

                WLL = Wc[dqm] + 0.5 * dW[dqm]
                WLR = Wc[dqp] - 0.5 * dW[dqp]

                ULL = np.zeros_like(WLL)
                ULR = np.zeros_like(WLR)
                self.to_conserved_vars(ULL, WLL)
                self.to_conserved_vars(ULR, WLR)

                RHOL, UL, PL = self.splitvars(WLL)
                RHOR, UR, PR = self.splitvars(WLR)
                AL = self.eos.querySoundSpeed(RHOL, PL)
                AR = self.eos.querySoundSpeed(RHOR, PR)
                A = np.maximum(np.abs(UL[k]) + AL, np.abs(UR[k]) + AR)

                F = - A * (ULR - ULL)
                F += self.compute_flux(WLL, axis=k)
                F += self.compute_flux(WLR, axis=k)
                F *= 0.5

                dF += np.diff(F, axis=k + 1) / self.grid.ds[k]

            if self.b_Cupy:
                dF = dF.get()
            self.U[qc] -= self.dt * dF

    def get_magsqr(self, a):
        return np.sum(a ** 2, axis=0)

    def splitvars(self, U):
        v1 = U[0]
        v2 = U[1:-1]
        v3 = U[-1]
        return v1, v2, v3

    def to_primitive_vars(self, U, W):
        if self.verbose > 3:
            print(" - Convert to primitive variable")
        rho, rhou, rhoe = self.splitvars(U)
        rho2, u, p = self.splitvars(W)
        rho2[:] = rho
        u[:] = rhou / rho
        p[:] = (rhoe - 0.5 * self.get_magsqr(rhou) / rho) * (self.eos.gamma - 1.0)

    def to_conserved_vars(self, U, W):
        if self.verbose > 3:
            print(" - Convert to conserved variable")
        rho, rhou, rhoe = self.splitvars(U)
        rho2, u, p = self.splitvars(W)
        rho[:] = rho2
        rhou[:] = rho * u
        rhoe[:] = 0.5 * rho * self.get_magsqr(u) + p / (self.eos.gamma - 1.0)

    def compute_flux(self, W, axis=-1):
        F = np.zeros_like(W)
        rho, u, p = self.splitvars(W)
        rho_flux, rhou_flux, rhoe_flux = self.splitvars(F)
        u_conv = u[axis]
        rho_flux[:] = u_conv * rho
        rhou_flux[:] = u_conv * rho * u
        rhou_flux[axis] += p
        rhoe_flux[:] = u_conv * (p + rho * (0.5 * self.get_magsqr(u) + p / rho / (self.eos.gamma - 1.0)))

        return F

    def dump_solution(self, n):
        if not self.b_dumpfile:
            return False
        self.refresh_buffer()
        rn = "result.%06d.tec" % (self.nti,)
        rho, u, p = self.splitvars(self.W)
        uvars = [("u_%d" % i, u[i].flatten(order='F')) for i in range(self.grid.ndims)]
        vars = (("rho", rho.flatten(order='F')), *uvars, ("p", p.flatten(order='F')))
        self.grid.dumpToTecplot(self.output_dir.joinpath(rn), vars)
        '''
        rn = "result%06d.pkl" % (self.nti,)
        with open(self.output_dir.joinpath(rn), 'wb') as handle:
            print("Dump solution -> " + "t=%4f r%06d" % (self.t, self.nti,))
            np.savez_compressed(handle, t=self.t, W=self.W)
        '''
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


def load_input(input_meta='euler.json'):
    with open(input_meta, ) as handle:
        return json.load(handle)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Opening JSON file
    meta = load_input('euler.json')
    # clear_output_dir(meta["output_dir"])

    eos_model = EOS_Base(input_json=meta["eos"])
    eos_model.showInfo()

    grid = UniformGrid()
    grid.size = tuple(meta["grid"]["size"])
    grid.rect = tuple(meta["grid"]["rect"])
    grid.n_decomposition = tuple(meta["grid"]["decompose"])
    grid.set_output_dir(meta["output_dir"])
    grid.b_Distribute = meta["grid"]["distributive"]
    grid.set_params()
    grid.create_stencil_op_collection()

    solver = EulerSolver(input_json=meta["solver"])
    solver.set_output_dir(meta["output_dir"])
    solver.set_grid(grid)
    solver.set_eos(eos_model)
    solver.set_params()
    solver.solve()
    # solver.dump_meta()
