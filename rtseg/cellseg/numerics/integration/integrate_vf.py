
import torch

from rtseg.cellseg.numerics.integration.solvers import (
    Euler, 
    Midpoint, 
    RungeKutta
)

SOLVERS = {
    "euler": Euler,
    "midpoint": Midpoint,
    "runge_kutta": RungeKutta
}

def ivp_solver(vf, init_values, dx, n_steps, solver = "euler", store_solutions: bool = True):
    """
    This function uses one of `SOLVERS` to solve the IVP given the inital
    values `init_values`, a step size `dx`, number of steps `n_steps`, and the
    continuous (through interpolation) vector field represented by `vf`.

    Args:
        vf (function): A function that can be built with `build_f` that converts
            some point `p` ((x, y)) to it's corresponding output (vector) in the
            vector field. 
        init_values (torch.Tensor): The initial values (semantic values) that
            will be clustered together under integration through the vector field.
            Of shape: (D, N).
        dx (float): The step size to use. I usually use a step size between 0
            and 1.
        n_steps (int): The number of steps each point will take through the
            vector field.
        solver (str): The solver to be used, you can find options in the above
            dictionary `SOLVERS` or `ivp_solvers.py`.
        
        store_solutions (bool): if True (default), the solver will store all the
            intermediate solutions on the way to clustering and will return a list
            whose last element is the final solution. Otherwise (False) returns
            a list with only one element.

    Returns: 
        torch.Tensor: List containing discretized trajectories. list[-1] will
        give the final solution obtained.

    """
    f_solver = SOLVERS[solver](vf)

    points, _ = f_solver.step(init_values, dx) # type: ignore

    solutions = [points]
    for i in range(n_steps - 1):
        points, _ = f_solver.step(points, dx) # type: ignore
        if store_solutions:
            solutions.append(points)
        else:
            solutions[0] = points

    return torch.stack(solutions)
