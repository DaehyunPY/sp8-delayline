from typing import Iterable, Mapping

from .units import in_milli_meter, in_nano_sec

__all__ = ('Read', 'Write')

try:
    from ROOT import TFile, TObject, TTree


    class Tree:
        def __init__(self, tree: TTree):
            self.__tree = tree

        def __len__(self):
            return self.__tree.GetEntries()

        def __iter__(self):
            for entry in self.__tree:
                yield self.__wrap(entry)

        def __getitem__(self, item):
            if isinstance(item, slice):
                start, stop, step = item.indices(len(self))
                return tuple(self[i] for i in range(start, stop, step))
            else:
                self.__tree.GetEntry(item)
                entry = self.__tree
                return self.__wrap(entry)

        @staticmethod
        def __wrap(entry: TTree):
            return {
                'ions': [
                    {'x': in_milli_meter(getattr(entry, 'IonX{:1d}'.format(i))),
                     'y': in_milli_meter(getattr(entry, 'IonY{:1d}'.format(i))),
                     't': in_nano_sec(getattr(entry, 'IonT{:1d}'.format(i))),
                     'flag': getattr(entry, 'IonFlag{:1d}'.format(i))}
                    for i in range(entry.IonNum)
                ],
                'electrons': [
                    {'x': in_milli_meter(getattr(entry, 'ElecX{:1d}'.format(i))),
                     'y': in_milli_meter(getattr(entry, 'ElecY{:1d}'.format(i))),
                     't': in_nano_sec(getattr(entry, 'ElecT{:1d}'.format(i))),
                     'flag': getattr(entry, 'ElecFlag{:1d}'.format(i))}
                    for i in range(entry.ElecNum)
                ]
            }


    class Read:
        def __init__(self, filename):
            self.__file = TFile(filename, 'READ')

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.__file.Close()
            if self.__file.IsOpen():
                raise IOError("TFile '{}' is not closed well!".format(self.__file))

        def __getitem__(self, item) -> Iterable[Mapping]:
            tree = getattr(self.__file, item)
            return Tree(tree)


    class Write:
        def __init__(self, filename):
            self.__file = TFile(filename, 'RECREATE')

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.__file.Write('', TObject.kOverwrite)
            self.__file.Close()
            if self.__file.IsOpen():
                raise IOError("TFile '{}' is not closed well!".format(self.__file))

        def __getattr__(self, item):
            return getattr(self.__file, item)

except ImportError:
    print('Module PyROOT is not imported!')
    TFile, TObject, TTree = None, None, None
    Read, Write = None, None
