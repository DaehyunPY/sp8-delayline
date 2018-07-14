from math import pi

from numba import jit


__all__ = ('in_degree', 'to_degree', 'in_nano_sec', 'to_nano_sec', 'in_femto_sec', 'to_femto_sec', 'in_milli_meter',
           'to_milli_meter', 'in_volt', 'to_volt', 'in_gauss', 'to_gauss', 'in_electron_volt', 'to_electron_volt',
           'in_atomic_mass', 'to_atomic_mass', 'unformat_to_num')


ma = 1.66053892173e-27  # atomic mass
c = 299792458  # speed of light
me = 9.1093829140e-31  # electron rest mass
e = 1.60217656535e-19  # elementary charge
hbar = 1.05457172647e-34
k = 8.9875517873681e9  # coulomb's constant
alpha = k * e ** 2 / hbar / c  # fine-structure constant
bohr = hbar / me / c / alpha
hartree = alpha ** 2 * me * c ** 2


@jit(nopython=True, nogil=True)
def in_degree(v):
    return v * pi / 180


@jit(nopython=True, nogil=True)
def to_degree(v):
    return v / pi * 180


@jit(nopython=True, nogil=True)
def in_nano_sec(v):
    return v * 1e-9 * hartree / hbar


@jit(nopython=True, nogil=True)
def to_nano_sec(v):
    return v / 1e-9 / hartree * hbar


@jit(nopython=True, nogil=True)
def to_femto_sec(v):
    return v / 1e-15 / hartree * hbar


@jit(nopython=True, nogil=True)
def in_femto_sec(v):
    return v * 1e-15 * hartree / hbar


@jit(nopython=True, nogil=True)
def in_milli_meter(v):
    return v * 1e-3 / bohr


@jit(nopython=True, nogil=True)
def to_milli_meter(v):
    return v / 1e-3 * bohr


@jit(nopython=True, nogil=True)
def in_volt(v):
    return v * e / hartree
in_electron_volt = in_volt


@jit(nopython=True, nogil=True)
def to_volt(v):
    return v / e * hartree
to_electron_volt = to_volt


@jit(nopython=True, nogil=True)
def in_gauss(v):
    return v * 1e-4 * e * bohr**2 / hbar


@jit(nopython=True, nogil=True)
def to_gauss(v):
    return v / 1e-4 / e / bohr**2 * hbar


@jit(nopython=True, nogil=True)
def in_atomic_mass(v):
    return v * ma / me


@jit(nopython=True, nogil=True)
def to_atomic_mass(v):
    return v / ma * me


@jit(nogil=True)
def unformat_to_num(inp: str) -> float:
    num_str, unit = inp.split()
    num = float(num_str)

    if unit == 'au':
        return num
    elif unit == 'deg':
        return in_degree(num)
    elif unit == 'ns':
        return in_nano_sec(num)
    elif unit == 'fs':
        return in_femto_sec(num)
    elif unit == 'mm':
        return in_milli_meter(num)
    elif unit == 'V':
        return in_volt(num)
    elif unit == 'G':
        return in_gauss(num)
    elif unit == 'eV':
        return in_electron_volt(num)
    elif unit == 'u':
        return in_atomic_mass(num)
    else:
        raise ValueError("Unit '{}' is not supported!".format(unit))
