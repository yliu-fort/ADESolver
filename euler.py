from ADESolver.euler import *

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
