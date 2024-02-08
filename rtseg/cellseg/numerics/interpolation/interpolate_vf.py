
from rtseg.cellseg.numerics.interpolation.functional import (
    bilinear_interpolation_batched,
    nearest_interpolation_batched,
    bilinear_interpolation,
    nearest_interpolation
)


def _vf_bilinear_batched(vector_field):
    def _vf(p):
        out = bilinear_interpolation_batched(vector_field, p)
        return out
    return _vf


def _vf_nearest_batched(vector_field):
    def _vf(p):
        out = nearest_interpolation_batched(vector_field, p)
        return out
    return _vf


def _vf_bilinear(vector_field):
    def _vf(p):
        out = bilinear_interpolation(vector_field[0], p)
        return out
    return _vf


def _vf_nearest(vector_field):
    def _vf(p):
        out = nearest_interpolation(vector_field[0], p)
        return out
    return _vf


vf_interpolators = {
    "bilinear_batched": _vf_bilinear_batched,
    "nearest_batched": _vf_nearest_batched,
    "bilinear": _vf_bilinear,
    "nearest": _vf_nearest
}


def interpolate_vf(vector_field, mode = "bilinear"):
    """
    Function that given a vector field, will return a function that you can call 
    with some points, to get back the values of the vector fields at those points.
    Args:
        Vector_field
    
    Returns:
        a function that takes point and closes over the provided vector_field, 
        so that you can repeatedly calculate the values of the vector field for 
        different points
    """
    return vf_interpolators[mode](vector_field)


