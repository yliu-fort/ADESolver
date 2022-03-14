from euler import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from pathlib import Path



def get_framenumber(input_meta):
    # Opening JSON file

    nfile = -1
    while True:
        rn = "result%06d.pkl" % (nfile + 1,)
        my_file = Path(input_meta["output_dir"]).joinpath(rn)
        if not my_file.is_file():
            break
        nfile += 1

    return nfile


def draw_density(input_meta):
    def f(i):
        rn = "result%06d.pkl" % (i,)
        #with open(Path(input_meta["output_dir"]).joinpath(rn), 'rb') as handle:
        #    solution = pickle.load(handle)
        solution = np.load(str(Path(input_meta["output_dir"]).joinpath(rn)), mmap_mode='r')
        field = solution["W"]

        slc = [slice(1)] * field.ndim
        slc[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc[-1] = slice(None)
        slc[-2] = slice(None)

        rho = np.squeeze(field[tuple(slc)])
        return rho

    def a():
        return 'Density (kg/m^3)'

    return f, a


def draw_pressure(input_meta):
    def f(i):
        rn = "result%06d.pkl" % (i,)
        #with open(Path(input_meta["output_dir"]).joinpath(rn), 'rb') as handle:
        #    solution = pickle.load(handle)
        #field = np.array(solution[1])
        solution = np.load(str(Path(input_meta["output_dir"]).joinpath(rn)), mmap_mode='r')
        field = solution["W"]

        slc = [slice(-1, None)] * field.ndim
        slc[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc[-1] = slice(None)
        slc[-2] = slice(None)
        return np.squeeze(field[tuple(slc)])

    def a():
        return 'Pressure (Pascal)'

    return f, a


def draw_machnumber(input_meta):
    def f(i):
        rn = "result%06d.pkl" % (i,)
        #with open(Path(input_meta["output_dir"]).joinpath(rn), 'rb') as handle:
        #    solution = pickle.load(handle)
        #field = np.array(solution[1])
        solution = np.load(str(Path(input_meta["output_dir"]).joinpath(rn)), mmap_mode='r')
        field = solution["rho"]

        slc1 = [slice(1)] * field.ndim
        slc1[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc1[-1], slc1[-2] = slice(None), slice(None)

        slc2 = [slice(-1, None)] * field.ndim
        slc2[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc2[-1], slc2[-2] = slice(None), slice(None)

        with open(Path(input_meta["output_dir"]).joinpath('meta.pkl'), 'rb') as handle:
            solver = pickle.load(handle)
        a = solver.eos.querySoundSpeed(np.squeeze(field[tuple(slc1)]), np.squeeze(field[tuple(slc2)]))

        slc3 = [slice(1, -1)] * field.ndim
        slc3[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc3[-1], slc3[-2] = slice(None), slice(None)

        umag = np.sqrt(solver.get_magsqr(field[tuple(slc3)]))

        return np.copy(np.squeeze(umag / a))

    def a():
        return 'Mach Number'

    return f, a


def draw_temperature(input_meta):
    def f(i):
        rn = "result%06d.pkl" % (i,)
        #with open(Path(input_meta["output_dir"]).joinpath(rn), 'rb') as handle:
        #    solution = pickle.load(handle)
        #field = np.array(solution[1])
        solution = np.load(str(Path(input_meta["output_dir"]).joinpath(rn)), mmap_mode='r')
        field = solution["W"]

        slc1 = [slice(1)] * field.ndim
        slc1[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc1[-1], slc1[-2] = slice(None), slice(None)

        slc2 = [slice(-1, None)] * field.ndim
        slc2[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
        slc2[-1], slc2[-2] = slice(None), slice(None)

        eos = EOS_Base(input_meta["eos"])

        T = eos.queryTemp(np.squeeze(field[tuple(slc1)]), np.squeeze(field[tuple(slc2)]))

        return np.copy(T)

    def a():
        return 'Temperature (K)'

    return f, a
