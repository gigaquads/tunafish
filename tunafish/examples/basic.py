from tunafish import FunctionTuner


def target(name: str, age: int, temperature: float, male: bool):
    """
    This is the "target" function whose output we're optimizing with
    respect to the inputs. In other words, we're finding the inputs which
    maximize fitness (the output of the function).
    """
    fitness = 0

    # consider name and male args as binary right/wrong
    fitness += int(name == 'Jeff')
    fitness += int(male)

    # penalize the more age and temp differs from 37 and 98.6
    fitness -= abs(age - 37)
    fitness -= abs(temperature - 98.6)

    return fitness
        

if __name__ == '__main__':
    # optional constraints on the argument search space
    options = {
        'age': {'min': 30, 'max': 50},
        'temperature': {'min': 91.0, 'max': 104},
        'name': {'enum': ['Jeff', 'John', 'Jimbob', 'Joffrey']},
    }

    # NOTE: statistics False in production by default
    tuner = FunctionTuner(statistics=True)

    # returns top individual from latest generation
    args = tuner.tune(target, options)

    print(args) 

    tuner.plot()