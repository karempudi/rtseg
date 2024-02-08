

# Different solvers for stepping through time to solve
# the initial value problems.

class Euler:
    """ Fixed Euler method. """
    def __init__(self, f):
        self.f = f 

    def step(self, points, dx):
        k1 = self.f(points)

        out = points + dx * k1

        return out, None


class Midpoint:
    """ Fixed Midpoint method. """
    def __init__(self, f):
        self.f = f

    def step(self, points, dx):
        k1 = self.f(points)
        k2 = self.f(points + dx * k1 / 2)

        out = points + dx * k2

        return out, None


class RungeKutta:
    """ Fixed 4th order Runge Kutta method. """
    def __init__(self, f):
        self.f = f 

    def step(self, points, dx):
        k1 = self.f(points)
        k2 = self.f(points + dx * k1 / 2)
        k3 = self.f(points + dx * k2 / 2)
        k4 = self.f(points + dx * k3)

        out = points + (dx / 6) * (k1 + (2 * k2) + (2 * k3) + k4)

        return out, None 
