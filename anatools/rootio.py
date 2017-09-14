from typing import Iterator, Mapping

from ROOT import TFile, TObject
from tqdm import tqdm

from .units import in_milli_meter, in_nano_sec

__all__ = ('Read', 'Write')


class Read:
    def __init__(self, filename):
        self.__file = TFile(filename, 'READ')

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


class Write:
    def __init__(self, filename):
        self.__file = TFile(filename, 'RECREATE')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__file.Write('', TObject.kOverwrite)
        self.__file.Close()

    def __getattr__(self, item):
        return getattr(self.__file, item)
