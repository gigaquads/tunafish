# Tunafish
## Autotune Python Functions
Tunafish is an argument autotuner for plain ol' Python functions. Under the hood, it uses genetic algorithms to determine which arguments maximize a function's return value (it's fitness). Tunafish eliminates the need to think in terms of genetic algorithms so you can focus more on what matters: your code.

To use Tunafish, you function must meet the following criteria:
1. Arguments *must* be annotated with primitives types, like `float`,
`int`, `str`, and `bool`, `typing.List`, etc.
2. The return value -- or, more generally, its output state --, *must* be
expressed as a single `float` (i.e. a fitness value).

## Automated Trading Example
Consider a function that places orders to buy and sell stocks. The inputs are `aggression`, which regulates the a minimum amount of time between placing orders, and `window`, which determines how far back the trading algorithm looks when deciding when buy or sell. The return value is simply the net gain or loss generated while trading.

### Define the Function
```python
from example_project import create_trader, load_historical_trading_data

trader = create_trader()
training_data = load_historical_trading_data(start, stop, interval)

def trade(aggression: float, window: int) -> float:
  gains = trader.trade(training_data, aggression, window)
  return gains  # AKA fitness
```

### Tune The Function
```python
from tunafish import FunctionTuner

tuner = FunctionTuner()
arguments = tuner.tune(trade, options={
  'aggression': {'min': 0.01, 'max': 1.0},
  'window': {'min': 5, 'max': 20}
})
```

## More Examples
Working examples can be found in `tunafish.examples`. The "basic" and "early_stopping" examples differ only in that "early_stopping" shows you how to control a bit more of the internals of the genetic algorithm. In particular, we tell it to exit the training loop early if we reach a fitness goal before all 500 epochs have run. Running these examples should generate a plot, showing convergence of fitness versus time.

![Max Fitness Per Epoch Graph](./docs/assets/fitness-per-epoch.png)

### Running Examples
Just do `python -m tunafish.examples.basic`!