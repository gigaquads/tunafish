import sys
import random
import os

from multiprocessing import Pool
from datetime import datetime
from typing import (
    Dict, List, Callable, Optional, Union, Sequence, Tuple
)

import deap.base
import deap.creator
import deap.tools
import numpy as np

from .parameter_specification import ParameterSpecification
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
        'mutation': 0.25,
    }

    def __init__(
        self,
        specs: List[ParameterSpecification],
        probabilities: Dict = DEFAULT_PROBABILITIES,
        statistics: bool = False,
        use_multiprocessing: bool = False,
        use_kwargs: bool = False,
    ):
        self.specs = specs
        self.keys = tuple(spec.name for spec in self.specs)
        self.probabilities = probabilities
        self.objective: Optional[Callable] = None
        self.are_statistics_enabled = statistics
        self.statistics = None
        self.use_kwargs = use_kwargs
        self.use_multiprocessing = use_multiprocessing
        self.pool = None
        self.epoch = 0

    def setup(self):
        """
        Initialize and register types and functions used internally by the
        Deap evolutionary computing framework.
        """
        toolbox = deap.base.Toolbox()

        # define geap internal sub-classes
        if not hasattr(deap.creator, 'FunctionTunerFitness'):
            deap.creator.create(
                'FunctionTunerFitness', deap.base.Fitness, weights=(1.0, )
            )
        if not hasattr(deap.creator, 'FunctionTunerIndividual'):
            deap.creator.create(
                'FunctionTunerIndividual', list,
                fitness=deap.creator.FunctionTunerFitness
            )

        # register our function that initializes each individual
        toolbox.register(
            'initializer', self.initializer
        )
        # tell Geap how to generate an individual with our initializer
        toolbox.register(
            'individual', deap.tools.initIterate,
            deap.creator.FunctionTunerIndividual, toolbox.initializer,
        )
        # create an initial population using the registered "individual" func
        toolbox.register(
            'population',
            deap.tools.initRepeat, list, toolbox.individual
        )
        # tell geap how to select the best performing individuals each epoch
        toolbox.register(
            'select', deap.tools.selTournament, tournsize=3
        )
        # tell geap how we want to perform genetic crossover
        toolbox.register(
            'mate', deap.tools.cxTwoPoint
        )
        # set the mutation operation
        toolbox.register(
            'mutate', deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.1
        )
        # set our fitness function
        toolbox.register(
            'evaluate', self.evaluate
        )

        if self.use_multiprocessing:
            pool = Pool(processes=max(1, os.cpu_count() or 1))
            toolbox.register('map', pool.map)

        return toolbox

    def fit(
        self,
        objective: Callable[..., float],
        population: Union[int, Sequence] = 25,
        epochs: int = 50,
        goal: float = None,
    ) -> List[List]:
        """
        Generate argument list (or lists) which maximize the fitness (a
        float) of the objective function. The objective function's output is
        expected to be its fitness.
        """
        toolbox = self.setup()

        self.objective = objective
        self.epoch = 0

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
            pop = toolbox.population(n=population)
        else:
            assert isinstance(population, (list, tuple, set))
            pop = list(population)

        is_goal_reached = False # only matters if goal is not None

        # evolve the initial population....
        for epoch in range(epochs):
            self.epoch = epoch
            print(f'Epoch {epoch}...') # TODO: replace with custom callback
            # generate next population from previous
            # using tournament selection
            offspring = list(
                map(
                    toolbox.clone,
                    toolbox.select(pop, len(pop))
                )
            )
            # apply crossovers
            for siblings in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.probabilities['crossover']:
                    toolbox.mate(*siblings)
                    for child in siblings:
                        del child.fitness.values

            # apply mutations
            for child in offspring:
                if random.random() < self.probabilities['mutation']:
                    toolbox.mutate(child)
                    del child.fitness.values

            # evaluate the fitness of each individual in need of it
            invalid_offspring = [x for x in offspring if not x.fitness.valid]
            fitness_values = toolbox.map(toolbox.evaluate, invalid_offspring)
            for child, fitness in zip(invalid_offspring, fitness_values):
                child.fitness.values = fitness
                if goal is not None and fitness[0] >= goal:
                    is_goal_reached = True

            for child in offspring:
                if goal is not None and child.fitness.values[0] >= goal:
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

    def evaluate(self, individual: List) -> Tuple[float]:
        """
        Translate genetic algorithm individual into arguments with which to
        call the objective function (that we're optimizing), call it, and
        return its fitness. This function is called internally by Geap.
        """
        args = Arguments.build(self.specs, individual)

        if self.use_kwargs:
            kwargs = dict(zip(self.keys, args))
            args.fitness = self.objective(**kwargs) or 0.0
        else:
            args.fitness = self.objective(*args) or 0.0

        return (args.fitness, )