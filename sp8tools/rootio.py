from itertools import count
from typing import Any

from tqdm import tqdm
from cytoolz import compose
from numba import jit

from .units import in_milli_meter, in_nano_sec

__all__ = ('Read', 'queries', 'events')

try:
    from ROOT import TFile, TObject, TTree

    class Read:
        def __init__(self, filename: str):
            self.__file = TFile(filename, 'READ')

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.__file.Close()

        def __getitem__(self, item: str) -> Any:
            return getattr(self.__file, item)

    @jit
    def partitions(chunk_size: int, n: int):
        if not n > 0:
            raise ValueError("Parameter 'n' must be 1 or bigger int!")
        for start in count(start=0, step=chunk_size):
            stop = start + chunk_size
            if stop < n:
                yield start, stop
            else:
                yield start, n
                break

    def queries(filename: str, treename: str, chunk_size: int = None):
        with Read(filename) as f:
            tree: TTree = f[treename]
            n = tree.GetEntries()
        for start, stop in partitions(chunk_size, n):
            yield {
                'filename': filename,
                'treename': treename,
                'start': start,
                'stop': stop
            }

    def events(filename: str, treename: str, start: int = 0, stop: int = None, step: int = 1):
        with Read(filename) as f:
            tree: TTree = f[treename]
            if stop is None:
                stop = tree.GetEntries()
            get = tree.GetEntry
            iattr = compose(tree.__getattr__, 'Ion{}'.format)
            eattr = compose(tree.__getattr__, 'Elec{}'.format)
            for entry in tqdm(range(start, stop, step)):
                get(entry)
                yield {
                    'ions': [
                        {'x': in_milli_meter(iattr('X' + str(i))),
                         'y': in_milli_meter(iattr('Y' + str(i))),
                         't': in_nano_sec(iattr('T' + str(i))),
                         'flag': iattr('Flag' + str(i))}
                        for i in range(iattr('Num'))
                    ],
                    'electrons': [
                        {'x': in_milli_meter(eattr('X' + str(i))),
                         'y': in_milli_meter(eattr('Y' + str(i))),
                         't': in_nano_sec(eattr('T' + str(i))),
                         'flag': eattr('Flag' + str(i))}
                        for i in range(eattr('Num'))
                    ]
                }

except ImportError:
    print('Module PyROOT is not imported!')
    TFile, TTree = None, None
    Read, queries, events = None, None, None
