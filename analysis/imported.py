from cytoolz import identity

try:
    from numba import jit
except ImportError:
    jit = identity

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
