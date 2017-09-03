from typing import Iterator, Mapping
from numpy import array, sin, cos
from cytoolz import curry
from ROOT import TFile
from .imported import jit, tqdm
from .units import in_milli_meter, in_nano_sec


@curry
@jit
def affine_transform(x, y, th=0, x0=0, y0=0, dx=1, dy=1, x1=0, y1=0):
    rot = array(((cos(th), sin(th)),
                 (-sin(th), cos(th))))
    return (rot @ (x, y) - (x0, y0)) * (dx, dy) + (x1, y1)


class Read:
    def __init__(self, filename):
        self.__file = TFile(filename, 'read')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__file.Close()

    def __getitem__(self, item) -> Iterator[Mapping]:
        tree = getattr(self.__file, item)
        for event in tqdm(tree, total=tree.GetEntries()):
            yield {'ion': tuple(
                       {'x': in_milli_meter(getattr(event, 'IonX{:1d}'.format(i))),
                        'y': in_milli_meter(getattr(event, 'IonY{:1d}'.format(i))),
                        't': in_nano_sec(getattr(event, 'IonT{:1d}'.format(i))),
                        'flag': getattr(event, 'IonFlag{:1d}'.format(i))} for i in range(event.IonNum)),
                   'electron': tuple(
                       {'x': in_milli_meter(getattr(event, 'ElecX{:1d}'.format(i))),
                        'y': in_milli_meter(getattr(event, 'ElecY{:1d}'.format(i))),
                        't': in_nano_sec(getattr(event, 'ElecT{:1d}'.format(i))),
                        'flag': getattr(event, 'ElecFlag{:1d}'.format(i))} for i in range(event.ElecNum))}
