import inspect

from typing import Dict, List, Callable, Optional, Union

import numpy as np

from .genetic import GeneticAlgorithm
from .spec import ParameterSpecification
from .arguments import Arguments


class FunctionTuner:
    """
    FunctionTuner determines which arguments to a function maximize its
    floating point return value, AKA its fitness.
    """

    def __init__(
        self,
        epochs: int = 256,
        population: Union[int, List[List[float]]] = 50,
        probabilities: Dict = GeneticAlgorithm.DEFAULT_PROBABILITIES,
        goal: Optional[float] = None,
        statistics: bool = False,
    ):
        self.specs = []
        self.epochs = max(1, epochs)
        self.goal = goal
        self.population = population or 50
        self.probabilities = probabilities
        self.are_statistics_enabled = statistics
        self.genetic_algorithm = None
        self.winner = None

    def tune(
        self,
        target: Callable[..., float],
        options: Optional[Dict] = None,
        many: bool = False,
    ) -> Union[Arguments, List[Arguments]]:
        """
        Use a genetic algorithm to determine the best input arguments to the
        target Python function, which maximize its return value -- that is,
        it's fitness.
        """
        # build parameter specs, used in transforming geap output into
        # corresponding argument lists
        signature = inspect.signature(target)
        for k, param in signature.parameters.items():
            spec = ParameterSpecification(param, options.get(k, {}))
            self.specs.append(spec)

        # run the genetic algorithm, returning optimial input args.
        # memoize them in case we want to resume tuning...
        self.genetic_algorithm = GeneticAlgorithm(
            specs=self.specs,
            probabilities=self.probabilities,
            statistics=self.are_statistics_enabled
        )
        self.population = self.genetic_algorithm.fit(
            target=target,
            population=self.population,
            epochs=self.epochs,
            goal=self.goal,
        )
        # convert lists of floats (the population) into argument tuples
        arg_lists = [Arguments.build(self.specs, x) for x in self.population]

        self.winner = arg_lists[-1]
        for args in reversed(arg_lists[:-1]):
            if args.fitness > self.winner.fitness:
                self.winner = args


        # return the first individual (an argument tuple) from the final
        # evolved population; otherwise, return the entire population.
        return arg_lists if many else arg_lists[0]

    def plot(self):
        assert self.genetic_algorithm
        assert self.genetic_algorithm.are_statistics_enabled

        from matplotlib import pyplot

        stats = self.genetic_algorithm.statistics
        x = np.arange(len(stats['fitness']))
        y = stats['fitness']

        fig, ax = pyplot.subplots(1, figsize=(14,6))

        fig.suptitle(
            f'Max Fitness Per Epoch'
            f'  (Fitness Goal: {self.goal:.2f})'
                if self.goal is not None else '',
            x=0.125, y=0.98, ha='left', fontsize=18
        )
        ax.set_title(
            f'Hall of Fame:  {tuple(self.winner) if self.winner else ""}',
            loc='left', fontsize=10, fontname='monospace',
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fitness')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
        
        ax.plot(x, y, linewidth=2.0)
        pyplot.show()