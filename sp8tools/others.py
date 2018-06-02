from numba import jit
from numpy import array, sin, cos, float64, ndarray


__all__ = ('rot_mat', 'affine_transform')


@jit(nopython=True, nogil=True)
def rot_mat(th: float) -> ndarray:
    return array(((cos(th), -sin(th)),
                  (sin(th), cos(th))), dtype=float64)


@jit(nopython=True, nogil=True)
def affine_transform(x: float, y: float, th: float=0, x0: float=0, y0: float=0, dx: float=1, dy: float=1,
                     x1: float=0, y1: float=0) -> ndarray:
    return ((rot_mat(th) @
             array((x, y), dtype=float64) - array((x0, y0), dtype=float64)) *
            array((dx, dy), dtype=float64) +
            array((x1, y1), dtype=float64))
