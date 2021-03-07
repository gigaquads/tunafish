import inspect

from typing import Dict, List, Callable, Optional, Union
from inspect import Parameter

import numpy as np

from .genetic_algorithm import GeneticAlgorithm
from .parameter_specification import ParameterSpecification
from .arguments import Arguments, KeywordArguments

VAR_POSITIONAL = Parameter.VAR_POSITIONAL
VAR_KEYWORD = Parameter.VAR_KEYWORD
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY


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
        use_multiprocessing: bool = False,
    ):
        self.specs = []
        self.epochs = max(1, epochs)
        self.goal = goal
        self.population = population or 50
        self.probabilities = probabilities
        self.are_statistics_enabled = statistics
        self.genetic_algorithm = None
        self.use_multiprocessing = use_multiprocessing
        self.winner = None

    @property
    def epoch(self) -> int:
        if self.genetic_algorithm:
            return self.genetic_algorithm.epoch
        return 0

    def tune(
        self,
        objective: Callable[..., float],
        options: Optional[Dict] = None,
    ) -> Union[KeywordArguments, Arguments, List]:
        """
        Use a genetic algorithm to determine the best input arguments to the
        objective Python function, which maximize its return value -- that is,
        it's fitness.
        """
        # build parameter specs, used in transforming geap output into
        # corresponding argument lists
        signature = inspect.signature(objective)
        params = list(signature.parameters.values())
        if len(params) == 1 and params[0].kind == VAR_KEYWORD:
            use_kwargs = True
            for k, v in options.items():
                annotation = v.get('dtype', float)
                param = Parameter(k, POSITIONAL_ONLY, annotation=annotation)
                spec = ParameterSpecification(param, options.get(k, {}))
                self.specs.append(spec)
        else:
            use_kwargs = False
            for k, param in signature.parameters.items():
                spec = ParameterSpecification(param, options.get(k, {}))
                self.specs.append(spec)

        # run the genetic algorithm, returning optimial input args.
        # memoize them in case we want to resume tuning...
        self.genetic_algorithm = GeneticAlgorithm(
            specs=self.specs,
            probabilities=self.probabilities,
            statistics=self.are_statistics_enabled,
            use_multiprocessing=self.use_multiprocessing,
            use_kwargs=use_kwargs
        )
        self.population = self.genetic_algorithm.fit(
            objective=objective,
            population=self.population,
            epochs=self.epochs,
            goal=self.goal,
        )
        # convert lists of floats (the population) into argument tuples
        if use_kwargs:
            results = [
                KeywordArguments.build(self.specs, x)
                for x in self.population
            ]
        else:
            results = [
                Arguments.build(self.specs, x)
                for x in self.population
            ]

        self.winner = results[-1]
        for x in reversed(results[:-1]):
            if x.fitness > self.winner.fitness:
                self.winner = x

        # return the first individual (an argument tuple/dict) from the final
        # evolved population; otherwise, return the entire population.
        return self.winner

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
            f'Hall of Fame:  {self.winner if self.winner else ""}',
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