from typing import Callable, Optional, NewType, NamedTuple


class Hit(NamedTuple):
    """
    Defines a set of detected time, position x and y
    """
    t: float
    x: float
    y: float


class AnalyzedHit(NamedTuple):
    """
    Defines a set of kinetic energy and momentum which are analyzed by a Model
    """
    ke: float
    pz: float
    px: float
    py: float
    # id_hit: Optional(int) = None
    # id_model: Optional(int) = None


Model = NewType('Model', Callable[[Hit], Optional[AnalyzedHit]])


class Accelerated(NamedTuple):
    accelerated_momentum: float
    flight_time: float


class Accelerator:
    __accelerate: Callable[..., Optional[Accelerated]]

    def __init__(self, accelerate: Callable[..., Optional[Accelerated]]):
        """
        Initialize an Accelerator. You might want to use this class as a decorator

        :param accelerate: a function(initial_momentum: float) -> Accelerated
        """
        self.__accelerate = accelerate

    def __call__(self, initial_momentum: float, **kwargs) -> Optional[Accelerated]:
        return self.__accelerate(initial_momentum, **kwargs)

    def __mul__(self, other: 'Accelerator') -> Optional['Accelerator']:
        """
        Compose two Accelerator instances. It is order sensitive! Where (self * other)(initial momentum),
        the accelerator 'self' will be called it after 'other' was.

        :param other: an Accelerator
        :return: composed Accelerator
        """

        def accelerate(initial_momentum, **kwargs) -> Optional[Accelerated]:
            acc0 = other(initial_momentum, **kwargs)
            if acc0 is None:
                return None
            acc1 = self(acc0.accelerated_momentum, **kwargs)
            if acc1 is None:
                return None
            return Accelerated(accelerated_momentum=acc1.accelerated_momentum,
                               flight_time=acc0.flight_time + acc1.flight_time)

        return Accelerator(accelerate)


def none_accelerator(length: float) -> Accelerator:
    if length <= 0:
        raise ValueError("Invalid argument 'length'!")

    @Accelerator
    def accelerator(initial_momentum: float, mass: float, **kwargs) -> Optional[Accelerated]:
        if initial_momentum <= 0:
            return None
        if mass <= 0:
            return None
        return Accelerated(accelerated_momentum=initial_momentum, flight_time=length / initial_momentum * mass)

    return accelerator


def simple_accelerator(length: float, electric_field: float) -> Accelerator:
    if length <= 0:
        raise ValueError("Invalid argument 'length'!")
    if electric_field == 0:
        raise ValueError("Invalid argument 'electric_field'!")

    @Accelerator
    def accelerator(initial_momentum: float, mass: float, charge: float, **kwargs) -> Optional[Accelerated]:
        if mass <= 0:
            return None
        if charge == 0:
            return None
        ke = initial_momentum ** 2 / 2 / mass + electric_field * charge * length
        if ke <= 0:
            return None
        p = (2 * ke * mass) ** 0.5
        t = (p - initial_momentum) / electric_field / charge
        return Accelerated(accelerated_momentum=p, flight_time=t)

    return accelerator
