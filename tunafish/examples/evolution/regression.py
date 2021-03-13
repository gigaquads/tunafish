import math
import random
import operator as op

from typing import (
    Any, Dict, Type, Callable, Text
)

import numpy as np

from numpy import ndarray as Array

from tunafish import AutomaticFunction


class SymbolicRegression(AutomaticFunction):

    def signature(self, x: float) -> float:
        raise NotImplemented

    def fitness(self, regression: Callable, x: Array) -> float:
        objective = self.context['objective']
        return np.mean(np.fromiter(
            ((regression(x_i) - objective(x_i))**2 for x_i in x), dtype=float
        ))

    def function_add(self, a: float, b: float) -> float:
        return a + b

    def function_sub(self, a: float, b: float) -> float:
        return a - b

    def function_mul(self, a: float, b: float) -> float:
        return a * b

    def function_div(self, a: float, b: float) -> float:
        return a / b if b else 1.0

    def function_neg(self, a: float) -> float:
        return -a

    def function_sin(self, a: float) -> float:
        return math.sin(a)

    def function_cos(self, a: float) -> float:
        return math.sin(a)

    def var_random_float(self) -> float:
        return float(random.randint(-1, 1))


if __name__ == '__main__':
    from matplotlib import pyplot as pp

    objective = lambda x: x**4 + x**3 + x**2 + x

    x = np.arange(-10, 10)

    regress = SymbolicRegression(
        generations=100,
        context={'objective': objective},
        max_tree_height=2,
        weight=-1.0,
    ).evolve(x)

    t = np.arange(-10, 10, 0.01)
    y = np.array([objective(x_i) for x_i in t], dtype=float)
    y_hat = np.array([regress(x_i) for x_i in t], dtype=float)

    print(str(regress.population[0]))

    pp.title('Symbolic Regression')
    pp.plot(t, y, label='Objective')
    pp.plot(t, y_hat, label='Regression')
    pp.show()