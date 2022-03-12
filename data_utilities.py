from main import load_input, UniformGrid
import cupy as cp
import numpy as np
import pickle
from pathlib import Path

from tvtk.api import tvtk, write_data
from render_functors import get_framenumber


def convertNPToVTKrect(input_meta):
    grid = UniformGrid()
    grid.size = tuple(input_meta["grid"]["size"])
    grid.rect = tuple(input_meta["grid"]["rect"])
    grid.set_params()
    if grid.ndims < 3:
        return

    nfile = get_framenumber(input_meta)
    while nfile >= 0:
        rn = "result%06d.pkl" % (nfile,)
        my_file = Path(input_meta["output_dir"]).joinpath(rn)
        if my_file.is_file():
            r = tvtk.RectilinearGrid()
            r.dimensions = grid.mesh[0].T.shape
            r.x_coordinates = grid.coords[-1]
            r.y_coordinates = grid.coords[-2]
            r.z_coordinates = grid.coords[-3]

            with open(Path(input_meta["output_dir"]).joinpath(rn), 'rb') as handle:
                solution = pickle.load(handle)
                field = np.array(solution[1])

                slc = [slice(1)] * field.ndim
                slc[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
                slc[-1] = slice(None)
                slc[-2] = slice(None)
                slc[-3] = slice(None)

                data = np.squeeze(field[tuple(slc)])  # scalar: nx x ny x nz

                r.point_data.scalars = data.ravel()
                r.point_data.scalars.name = 'rho'

                # add second point data field
                slc[0] = slice(1, -1)
                data = field[tuple(slc)]  # scalar: nx x ny x nz

                r.point_data.vectors = np.moveaxis(np.array([np.squeeze(data[i]).ravel() for i in range(3)]), 0, -1)
                r.point_data.vectors.name = 'U'

                # add second point data field
                slc[0] = slice(-1, None)
                data = np.squeeze(field[tuple(slc)])  # scalar: nx x ny x nz

                r.point_data.add_array(data.ravel())
                r.point_data.get_array(2).name = 'p'
                r.point_data.update()

                ro = "result%06d.vtk" % (nfile,)
                write_data(r, str(Path(input_meta["output_dir"]).joinpath(ro)))
                del field, solution
                print("Write result to VTK %06d" % nfile)

        nfile -= 1

def convertNPToVTKrectDecomposed(solver, field, nfile):
    if solver.grid.ndims < 3:
        return

    r = tvtk.RectilinearGrid()
    r.dimensions = solver.grid.mesh[0].T.shape
    r.x_coordinates = solver.grid.coords[-1]
    r.y_coordinates = solver.grid.coords[-2]
    r.z_coordinates = solver.grid.coords[-3]

    # add density field
    slc = [slice(1)] * field.ndim
    slc[1:] = [slice(x - 1, x) for x in [int(np.round(x / 2)) for x in field.shape[1:]]]
    slc[-1] = slice(None)
    slc[-2] = slice(None)
    slc[-3] = slice(None)

    data = np.squeeze(field[tuple(slc)])  # scalar: nx x ny x nz

    r.point_data.scalars = data.ravel()
    r.point_data.scalars.name = 'rho'

    # add velocity field
    slc[0] = slice(1, -1)
    data = field[tuple(slc)]  # scalar: nx x ny x nz

    r.point_data.vectors = np.moveaxis(np.array([np.squeeze(data[i]).ravel() for i in range(3)]), 0, -1)
    r.point_data.vectors.name = 'U'

    # add pressure field
    slc[0] = slice(-1, None)
    data = np.squeeze(field[tuple(slc)])  # scalar: nx x ny x nz

    r.point_data.add_array(data.ravel())
    r.point_data.get_array(2).name = 'p'
    r.point_data.update()

    ro = "result%06d.vtk" % (nfile,)
    write_data(r, str(Path(solver.output_dir).joinpath(ro)))

    print("Write result to VTK %06d" % nfile)

def convertCUfileToNP(input_meta):
    nfile = -1
    while True:
        rn = "result%06d.pkl" % (nfile + 1,)
        my_file = Path(input_meta["output_dir"]).joinpath(rn)
        if not my_file.is_file():
            break
        else:
            nfile += 1
            with open(Path(input_meta["output_dir"]).joinpath(rn), 'rb') as handle:
                solution = pickle.load(handle)
                if isinstance(solution[1], cp.ndarray):
                    with open(Path(input_meta["output_dir"]).joinpath(rn), 'wb') as inhandle:
                        pickle.dump((solution[0], cp.asnumpy(solution[1])), inhandle, protocol=pickle.HIGHEST_PROTOCOL)
                        print("Convert result %06d" % nfile)


if __name__ == '__main__':
    convertCUfileToNP(load_input('input.json'))
    #convertNPToVTKrect(load_input('input.json'))
