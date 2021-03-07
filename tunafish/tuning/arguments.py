from typing import List, Optional


class Arguments(tuple):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fitness: float = 0
        self.individual: Optional[List[float]] = None

    @classmethod
    def build(cls, specs, individual: List[float]) -> 'Arguments':
        """
        Convert raw genetic algorithm output floats into function argument
        values.
        """
        return cls([s.fit(x) for s, x in zip(specs, individual)])


class KeywordArguments(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness: float = 0
        self.individual: Optional[List[float]] = None

    @classmethod
    def build(cls, specs, individual: List[float]) -> 'KeywordArguments':
        """
        Convert raw genetic algorithm output floats into function argument
        values.
        """
        return cls({s.name: s.fit(x) for s, x in zip(specs, individual)})