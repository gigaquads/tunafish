import sys
import random
import inspect

from datetime import datetime
from typing import (
    Dict, List, Callable, Optional, Union, Sequence
)

import deap.base
import deap.creator
import deap.tools
import numpy as np

from .spec import ParameterSpecification
from .arguments import Arguments


class GeneticAlgorithm:
    """
    This GeneticAlgorithm class is just a convenient encapsulation of the
    more awkward Geap evolutionary algorithm framework. The algorithm simply
    generates populations of lists of floats. Each float gets
    fitted/translated into a corresponding argument value expected by the
    Python function being optimized.
    """

    DEFAULT_PROBABILITIES: Dict = {
        'crossover': 0.5,
        'mutation': 0.2
    }

    def __init__(
        self,
        specs: List[ParameterSpecification],
        probabilities: Dict = DEFAULT_PROBABILITIES,
        statistics: bool = False,
    ):
        self.specs = specs
        self.probabilities = probabilities
        self.toolbox = deap.base.Toolbox()
        self.target: Optional[Callable] = None
        self.are_statistics_enabled = statistics
        self.statistics = None
        self.setup()

    def setup(self):
        """
        Initialize and register types and functions used internally by the
        Deap evolutionary computing framework.
        """
        # define geap internal sub-classes
        deap.creator.create(
            'BasicFitness', deap.base.Fitness, weights=(1.0, )
        )
        deap.creator.create(
            'Individual', list, fitness=deap.creator.BasicFitness
        )
        # register our function that initializes each individual
        self.toolbox.register(
            'initializer', self.initializer
        )
        # tell Geap how to generate an individual with our initializer
        self.toolbox.register(
            'individual', deap.tools.initIterate, deap.creator.Individual,
            self.toolbox.initializer,
        )
        # create an initial population using the registered "individual" func
        self.toolbox.register(
            'population',
            deap.tools.initRepeat, list, self.toolbox.individual
        )
        # tell geap how to select the best performing individuals each epoch
        self.toolbox.register(
            'select', deap.tools.selTournament, tournsize=3
        )
        # tell geap how we want to perform genetic crossover
        self.toolbox.register(
            'mate', deap.tools.cxTwoPoint
        )
        # set the mutation operation
        self.toolbox.register(
            'mutate', deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.1
        )
        # set our fitness function
        self.toolbox.register(
            'evaluate', lambda individual: (self.evaluate(individual), )
        )

    def fit(
        self,
        target: Callable[..., float],
        population: Union[int, Sequence] = 25,
        epochs: int = 50,
        goal: float = None,
    ) -> List[List]:
        """
        Generate argument list (or lists) which maximize the fitness (a
        float) of the target function. The target function's output is
        expected to be its fitness.
        """
        self.target = target

        # initialize a new statistics object
        if self.are_statistics_enabled:
            self.statistics = {
                'fitness': [],
                'winners': [],
                'start': datetime.now(),
                'stop': None,  # : datetime
                'time': None,  # : timedelta
            }

        # generate initial population
        if isinstance(population, int):
            pop = self.toolbox.population(n=population)
        else:
            assert isinstance(population, (list, tuple, set))
            pop = list(population)

        is_goal_reached = False # only matters if goal is not None

        # evolve the initial population....
        for epoch in range(epochs):
            # generate next population from previous
            # using tournament selection
            offspring = list(
                map(
                    self.toolbox.clone,
                    self.toolbox.select(pop, len(pop))
                )
            )
            # apply crossovers
            for siblings in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.probabilities['crossover']:
                    self.toolbox.mate(*siblings)
                    for child in siblings:
                        del child.fitness.values

            # apply mutations
            for child in offspring:
                if random.random() < self.probabilities['mutation']:
                    self.toolbox.mutate(child)
                    del child.fitness.values

            # evaluate the fitness of each individual in need of it
            invalid_offspring = [x for x in offspring if not x.fitness.valid]
            fitness_values = map(self.toolbox.evaluate, invalid_offspring)
            for child, fitness in zip(invalid_offspring, fitness_values):
                child.fitness.values = fitness
                if goal is not None and fitness[0] >= goal:
                    is_goal_reached = True

            # replace last generation with the next
            pop = offspring

            # update per-epoch statistics
            if self.are_statistics_enabled:
                # keep track of each epoch's max fitness
                max_fitness = -1 * sys.maxsize
                winner = None
                for child in offspring:
                    fitness = child.fitness.values[0]
                    if fitness > max_fitness:
                        max_fitness = fitness
                        winner = child
                self.statistics['fitness'].append(max_fitness)
                self.statistics['winners'].append(winner)

            if is_goal_reached:
                break

        # update statistics
        if self.are_statistics_enabled:
            self.statistics['fitness'] = np.array(self.statistics['fitness'])
            self.statistics['winners'] = np.array(self.statistics['winners'])
            self.statistics['stop'] = datetime.now()
            self.statistics['time'] = (
                self.statistics['stop'] - self.statistics['start']
            )

        return pop

    def initializer(self) -> List:
        """
        This function generates a new individual for our genetic algorithm.
        This function is called internally by Geap.
        """
        arg_list = []  # <- returned 
        for spec in self.specs:
            if spec.sequence_dtype is not None:
                raise NotImplementedError(
                    'TODO: use a separate GA for this list argument'
                )
            else:
                # x is our value to append to values list
                x = random.random() * 100.0  # default float value

                # compute x (override its current default value)
                if spec.enum_values:
                    x = random.randrange(len(spec.enum_values))
                elif spec.dtype is bool:
                    x = random.random()
                else:
                    x_min = spec.min_value
                    x_max = spec.max_value
                    if spec.dtype is int:
                        x = int(x_min + random.random() * (x_max - x_min))
                    if spec.dtype is float:
                        x = x_min + random.random() * (x_max - x_min)

                arg_list.append(x)

        return arg_list

    def evaluate(self, individual: List) -> float:
        """
        Translate genetic algorithm individual into arguments with which to
        call the target function (that we're optimizing), call it, and
        return its fitness. This function is called internally by Geap.
        """
        args = Arguments.build(self.specs, individual)
        args.fitness = self.target(*args) or 0.0
        return args.fitness or 0.0