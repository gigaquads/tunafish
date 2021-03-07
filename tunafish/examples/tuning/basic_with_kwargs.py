from appyratus.utils.time_utils import TimeUtils

from tunafish import FunctionTuner


def objective(**kwargs):
    """
    This is the "objective" function whose output we're optimizing with
    respect to the inputs. In other words, we're finding the inputs which
    maximize fitness (the output of the function).
    """
    fitness = 0

    # consider name and male args as binary right/wrong
    fitness += int(kwargs['name'] == 'Jeff')
    fitness += int(kwargs['male'])

    # penalize the more age and temp differs from 37 and 98.6
    fitness -= abs(kwargs['age'] - 37)
    fitness -= abs(kwargs['temperature'] - 98.6)

    return fitness
        

if __name__ == '__main__':
    # optional constraints on the argument search space
    options = {
        'age': {'dtype': int, 'min': 30, 'max': 50},
        'name': {'dtype': str, 'enum': ['Jeff', 'John', 'Jimbob', 'Joffrey']},
        'temperature': {'dtype': float, 'min': 91.0, 'max': 104},
        'male': {'dtype': bool}
    }

    tuner = FunctionTuner(epochs=256, statistics=True, use_multiprocessing=False)
    kwargs, time = TimeUtils.timed(lambda: tuner.tune(objective, options))

    print(f'Runtime: {time.total_seconds():.2f}s')
    print(f'Best Kwargs:', kwargs)

    tuner.plot()