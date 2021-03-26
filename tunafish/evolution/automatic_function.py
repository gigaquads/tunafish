import operator
import inspect
import pickle

from multiprocessing import Pool
from inspect import signature
from typing import (
    Any, Dict, Callable, Union, Tuple, Text, Optional
)

import deap.base
import deap.creator
import deap.tools
import deap.gp
import deap.algorithms
import numpy as np

from deap.gp import PrimitiveSetTyped

from .util import (
    clamp, get_required_parameters, is_parameter_required
)


class AutomaticFunction:
    def __init__(
        self,
        generations: int = 64,
        population: int = 256,
        min_tree_height: int = 1,
        max_tree_height: int = 5,
        tournament_size: int = 3,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.1,
        weights: Optional[Tuple] = (1.0, ),
        weight: Optional[float] = None,
        verbose: bool = True,
        context: Dict = None,
        meta: dict = None,
    ):
        self.class_name = type(self).__name__
        self.n_generations = clamp(generations, 2, 100000)
        self.n_population = clamp(population, 2, 5000)
        self.min_tree_height = clamp(min_tree_height, 1, 17)
        self.max_tree_height = clamp(max_tree_height, min_tree_height, 17)
        self.tournament_size = clamp(tournament_size, 1, population // 2)
        self.crossover_rate = clamp(crossover_rate, 0.01, 1.0)
        self.mutation_rate = clamp(mutation_rate, 0.01, 1.0)
        self.verbose = verbose
        self.context = context or {}
        self.meta = meta or {}

        # vars set by self.evolve:
        self.logbook = None
        self.population = None
        self.expressions = None
        self.winner = None
        self.winners = []
        self.winner_expression = None
        self.winner_expressions = []

        if weight is not None:
            self.weights = (weight, )
        else:
            assert weights
            if isinstance(weights, (list, np.ndarray)):
                self.weights = tuple(weights)
            else:
                self.weights = weights

        self.unfit = tuple(
            -np.inf if w >= 0 else np.inf
            for w in self.weights
        )

        # input_items are (arg_name, dtype) tuples...
        sig = inspect.signature(self.stub)
        input_items = tuple(
            (k, v.annotation)
            for k, v in list(sig.parameters.items())
            if is_parameter_required(v)
        )
        # create container for functions and terminals used by automatic_function
        self.primitives = PrimitiveSetTyped(
            self.class_name, [item[1] for item in input_items],
            sig.return_annotation
        )
        # rename input arguments from geap defaults to
        # the keys in the self.inputs dict.
        self.primitives.renameArguments(**{
            f'ARG{i}': item[0] for i, item in enumerate(input_items)
        })

        # add self.functions to primitives
        functions = self.functions
        if not functions:
            functions = {}
            for k in dir(self):
                if k.startswith('function_'):
                    func = getattr(self, k)
                    if callable(func):
                        name = k[9:].upper()
                        functions[name] = func

        for name, func in functions.items():
            self.register_function(func, name=name)

        # add constants to primitives
        constants = self.constants or {
            k[6:]: getattr(self, k) for k in dir(self)
            if k.startswith('const_')
        }
        for name, func in constants.items():
            value = func()
            self.primitives.addTerminal(value, type(value), name=name)

        # add variables (variables are callbacks that execute at runtime to
        # generate a dynamic value). Deap calls these "ephemeral constants".
        variables = self.variables or {
            k[4:]: getattr(self, k) for k in dir(self)
            if k.startswith('var_')
        }
        for name, func in variables.items():
            if name == 'random':
                # Deap breaks if the function's name is "random"
                name = 'random_'
            sig = signature(func)
            ret_type = sig.return_annotation
            # import ipdb; ipdb.set_trace()
            self.primitives.addEphemeralConstant(name, func, ret_type)

    def __call__(self, *args, **kwargs):
        return self.winner(*args, *kwargs) if self.winner else None

    def register_function(self, func: Callable, name: str = None):
        name = name or func.__name__
        sig = inspect.signature(func)
        params = get_required_parameters(sig)
        in_types = [p.annotation for p in params.values()]
        ret_type = sig.return_annotation
        self.primitives.addPrimitive(
            func, in_types=in_types, ret_type=ret_type, name=name
        )

    def stub(self, x: float) -> float:
        raise NotImplementedError()

    def setup(self, args=None, kwargs=None):
        self.context['args'] = args or tuple()
        self.context['kwargs'] = kwargs or {}

        # register Deap internal classes
        if not hasattr(deap.creator, f'{self.class_name}Fitness'):
            deap.creator.create(
                f'{self.class_name}Fitness', deap.base.Fitness,
                weights=self.weights,
            )
        deap.creator.create(
            f'{self.class_name}Individual', deap.gp.PrimitiveTree,
            fitness=getattr(deap.creator, f'{self.class_name}Fitness')
        )
        # register Deap internal callbacks
        toolbox = deap.base.Toolbox()
        toolbox.register('expr',
            deap.gp.genHalfAndHalf, pset=self.primitives,
            min_=self.min_tree_height, max_=self.max_tree_height
        )
        toolbox.register('individual',
            deap.tools.initIterate,
            getattr(deap.creator, f'{self.class_name}Individual'),
            toolbox.expr
        )
        toolbox.register(
            'population', deap.tools.initRepeat, list, toolbox.individual
        )
        toolbox.register(
            'compile', deap.gp.compile, pset=self.primitives
        )
        toolbox.register(
            'select', deap.tools.selTournament,
            tournsize=self.tournament_size
        )
        toolbox.register(
            'mate', deap.gp.cxOnePoint
        )
        toolbox.register(
            'expr_mut', deap.gp.genFull, min_=0, max_=self.max_tree_height
        )
        toolbox.register(
            'mutate', deap.gp.mutUniform, expr=toolbox.expr_mut,
            pset=self.primitives
        )
        toolbox.register(
            'evaluate', self.compile_and_evaluate, args=args, kwargs=kwargs
        )
        # set hard limit for total tree size to 17, as recommended by Koza
        toolbox.decorate(
            'mate', deap.gp.staticLimit(
                key=operator.attrgetter('height'), max_value=17
            )
        )
        toolbox.decorate(
            'mutate', deap.gp.staticLimit(
                key=operator.attrgetter('height'), max_value=17
            )
        )
        return toolbox

    def evolve(self, *args, **kwargs) -> 'AutomaticFunction':
        toolbox = self.setup(args, kwargs)

        # configure statistics for evolve loop
        stats_fit = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = deap.tools.Statistics(len)
        stats = deap.tools.MultiStatistics(
            fitness=stats_fit, size=stats_size
        )
        stats.register('avg', np.nanmean)
        stats.register('std', np.nanstd)
        stats.register('min', np.nanmin)
        stats.register('max', np.nanmax)

        # initialize generation 0
        pop = toolbox.population(n=self.n_population)

        # kick off the evolve loop...
        hof = deap.tools.HallOfFame(10)
        final_pop, self.logbook = deap.algorithms.eaSimple(
            pop, toolbox, self.crossover_rate, self.mutation_rate,
            self.n_generations, stats=stats, halloffame=hof,
            verbose=self.verbose
        )

        # memoize the final population in the form of compile funcs
        self.population = final_pop
        self.expressions = [
            deap.gp.compile(individual, self.primitives)
            for individual in final_pop
        ]
        # set the most fit individual (evolved function)
        self.winner = deap.gp.compile(hof[-1], self.primitives)
        self.winner_expression = hof[-1]
        self.winner_expressions = list(hof)
        self.winners = [
            deap.gp.compile(expr, self.primitives)
            for expr in hof
        ]

        return self

    def compile_and_evaluate(self, individual, args, kwargs) -> Tuple:
        evolved_func = deap.gp.compile(individual, self.primitives)
        fitness = self.fitness(evolved_func, *args, **kwargs)
        if isinstance(fitness, (int, float, np.number)):
            return (float(fitness), )
        elif isinstance(fitness, (list, np.ndarray)):
            return tuple(fitness)
        else:
            assert isinstance(fitness, tuple)
            return fitness

    def compile(self, individual) -> Callable:
        return deap.gp.compile(individual, self.primitives)

    @property
    def functions(self) -> Dict[Text, Callable]:
        return {}

    @property
    def constants(self) -> Dict[Text, Any]:
        return {}

    @property
    def variables(self) -> Dict[Text, Callable]:
        return {}

    @property
    def source(self) -> Optional[str]:
        """
        Return source code string of the evolution winner.
        """
        return (
            str(self.winner_expression) if self.winner_expression else None
        )

    def fitness(self, func: Callable, *args, **kwargs):
        """
        Returns any of the following:
        float, int, list, tuple, np.ndarray, np.number
        """
        raise NotImplementedError()

    def pickle(self, path: str, meta: dict = None, *args, **kwargs) -> dict:
        if meta:
            self.meta.update(meta)
        data = {
            # NOTE: we must store some fields as pickles inside the main pickle
            # in order to accomodate limitations in unpickling Deap components.
            'population': pickle.dumps(self.population),
            'winners': pickle.dumps(self.winner_expressions),

            'logbook': self.logbook,
            'kwargs': {
                'generations': self.n_generations,
                'population': self.n_population,
                'min_tree_height': self.min_tree_height,
                'max_tree_height': self.max_tree_height,
                'tournament_size': self.tournament_size,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'weights': self.weights,
                'verbose': self.verbose,
                'meta': self.meta,
            }
        }
        with open(path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
            return data

    @classmethod
    def unpickle(cls, path: str) -> 'AutomaticFunction':
        with open(path, 'rb') as pickle_file:
            state = pickle.load(pickle_file)
            kwargs = state['kwargs']

        func = cls(**kwargs)
        func.setup()

        # func.setup must be called before Deap components are unpickled. This
        # forces us to store the "population" as a pickle inside the pickle
        # itself -- so that we can unpickled it after calling setup.
        population = pickle.loads(state['population'])
        winner_expressions = pickle.loads(state['winners'])

        expressions = [func.compile(x) for x in population]
        winners = [func.compile(x) for x in winner_expressions]

        func.logbook = state['logbook']
        func.population = population
        func.expressions = expressions
        func.winner_expressions = winner_expressions
        func.winner_expression = winner_expressions[-1]
        func.winners = winners
        func.winner = winners[-1]

        return func
