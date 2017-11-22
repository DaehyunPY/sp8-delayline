from cytoolz import pipe
from cytoolz.curried import filter, flip
from numpy import nan, arctan2, arccos, average, sum, isfinite

from .units import as_milli_meter, as_nano_sec, as_electron_volt, as_degree, as_atomic_mass

__all__ = ('Object', 'Objects')


class Object:
    def __init__(self, x, y, t, ke=nan, px=nan, py=nan, pz=nan):
        self.__x = x
        self.__y = y
        self.__t = t
        self.__ke = ke
        self.__px = px
        self.__py = py
        self.__pz = pz

    @property
    def mass_(self):
        return self.pr_ ** 2 / self.ke_ / 2

    @property
    def mass(self):
        return as_atomic_mass(self.mass_)

    @property
    def weight(self):
        return self.mass_ / self.t_

    @property
    def x_(self):
        return self.__x

    @property
    def y_(self):
        return self.__y

    @property
    def r_(self):
        return (self.x_ ** 2 + self.y_ ** 2) ** 0.5

    @property
    def t_(self):
        return self.__t

    @property
    def x(self):
        return as_milli_meter(self.x_)

    @property
    def y(self):
        return as_milli_meter(self.y_)

    @property
    def r(self):
        return as_milli_meter(self.r_)

    @property
    def dir_xy_(self):
        return arctan2(self.y_, self.x_)

    @property
    def dir_xy(self):
        return as_degree(self.dir_xy_)

    @property
    def t(self):
        return as_nano_sec(self.t_)

    @property
    def px_(self):
        return self.__px

    @property
    def py_(self):
        return self.__py

    @property
    def pz_(self):
        return self.__pz

    @property
    def pxy_(self):
        return (self.px_ ** 2 + self.py_ ** 2) ** 0.5

    @property
    def pyz_(self):
        return (self.py_ ** 2 + self.pz_ ** 2) ** 0.5

    @property
    def pzx_(self):
        return (self.pz_ ** 2 + self.px_ ** 2) ** 0.5

    @property
    def pr_(self):
        return (self.px_ ** 2 + self.py_ ** 2 + self.pz_ ** 2) ** 0.5

    @property
    def px(self):
        return self.px_

    @property
    def py(self):
        return self.py_

    @property
    def pz(self):
        return self.pz_

    @property
    def pxy(self):
        return self.pxy_

    @property
    def pyz(self):
        return self.pyz_

    @property
    def pzx(self):
        return self.pzx_

    @property
    def pr(self):
        return self.pr_

    @property
    def cos_dir_pz_(self):
        return self.pz_ / self.pr_

    @property
    def dir_pz_(self):
        return arccos(self.cos_dir_pz_)

    @property
    def dir_pxy_(self):
        return arctan2(self.py_, self.px_)

    @property
    def dir_pyz_(self):
        return arctan2(self.pz_, self.py_)

    @property
    def dir_pzx_(self):
        return arctan2(self.px_, self.pz_)

    @property
    def cos_dir_pz(self):
        return self.cos_dir_pz_

    @property
    def dir_pz(self):
        return as_degree(self.dir_pz)

    @property
    def dir_pxy(self):
        return as_degree(self.dir_pxy_)

    @property
    def dir_pyz(self):
        return as_degree(self.dir_pyz_)

    @property
    def dir_pzx(self):
        return as_degree(self.dir_pzx_)

    @property
    def ke_(self):
        return self.__ke

    @property
    def ke(self):
        return as_electron_volt(self.ke_)

    @property
    def has_momentum(self):
        return isfinite(self.ke_)


class Objects(Object):
    def __init__(self, *objects: Object):
        self.__objects = tuple(objects)
        self.__having_momentum = pipe(self, filter(flip(getattr, 'has_momentum')), tuple)
        if len(self.having_momentum) == 0:
            super().__init__(x=nan, y=nan, t=nan)
            self.__weight = 0
        elif len(self.having_momentum) == 1:
            o = self.having_momentum[0]
            super().__init__(x=o.x_, y=o.y_, t=o.t_, ke=o.ke_, px=o.px_, py=o.py_, pz=o.pz_)
            self.__weight = o.weight
        else:
            x = tuple(o.x_ for o in self.having_momentum)
            y = tuple(o.x_ for o in self.having_momentum)
            w = tuple(o.weight for o in self.having_momentum)
            ke = tuple(o.ke_ for o in self.having_momentum)
            px = tuple(o.px_ for o in self.having_momentum)
            py = tuple(o.py_ for o in self.having_momentum)
            pz = tuple(o.pz_ for o in self.having_momentum)
            super().__init__(x=average(x, weights=w), y=average(y, weights=w), t=nan,
                             ke=sum(ke), px=sum(px), py=sum(py), pz=sum(pz))
            self.__weight = sum(w)

    def __len__(self):
        return len(self.__objects)

    def __getitem__(self, item):
        return self.__objects[item]

    def __iter__(self):
        return iter(self.__objects)

    @property
    def having_momentum(self):
        return self.__having_momentum

    @property
    def weight(self):
        return self.__weight

    @property
    def t_(self):
        return self.mass_ / self.weight
