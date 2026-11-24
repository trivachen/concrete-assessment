from functools import reduce
from typing import Callable


def compose(*fns) -> Callable:
    return reduce(lambda f, g: lambda x: f(g(x)), fns)
