from .imported import jit

ma = 1.66053892173e-27  # atomic mass
c = 299792458  # speed of light
me = 9.1093829140e-31  # electron rest mass
e = 1.60217656535e-19  # elementary charge
hbar = 1.05457172647e-34
k = 8.9875517873681e9  # coulomb's constant
alpha = k * e ** 2 / hbar / c  # fine-structure constant
bohr = hbar / me / c / alpha
hartree = alpha ** 2 * me * c ** 2


@jit
def in_nano_sec(v):
    return v * 1e-9 * hartree / hbar


@jit
def as_nano_sec(v):
    return v / 1e-9 / hartree * hbar


@jit
def in_milli_meter(v):
    return v * 1e-3 / bohr


@jit
def as_milli_meter(v):
    return v / 1e-3 * bohr


@jit
def in_volt(v):
    return v * e / hartree


@jit
def as_volt(v):
    return v / e * hartree


@jit
def in_degree(v):
    return v * pi / 180


@jit
def as_degree(v):
    return v / pi * 180


@jit
def in_gauss(v):
    return v * 1e-4 * e * bohr**2 / hbar


@jit
def as_gauss(v):
    return v / 1e-4 / e / bohr**2 * hbar


@jit
def in_electron_volt(v):
    return v * e / hartree


@jit
def as_electron_volt(v):
    return v / e * hartree


@jit
def in_atomic_mass(v):
    return v * ma / me


@jit
def as_atomic_mass(v):
    return v / ma * me
