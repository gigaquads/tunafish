from inspect import Parameter
from typing import (
    Dict, Any, Type, Tuple, List, Text,
    Set, Optional
)


class ParameterSpecification:
    """
    The ParameterSpecification contains data necessary to convert the output
    of a genetic algorithm into an int, float, string, bool or other datatype
    expected by the function who owns the parameter.

    The genetic algorithm generates a float for each argument expected by the
    parameter's owner function. Each float must be converted into a
    corresponding int, float, bool, string or other datatype expected by the
    function.

    Note that, for each non-scalar parameter -- that is, for lists, tuple,
    and sets -- the sequence is generated by a separate genetic algorithm,
    forming a tree of genetic algorithms.
    """

    def __init__(self, param: Parameter, options: Dict):
        self.param = param
        self.options = options
        self.dtype, self.sequence_dtype = self._infer_type(param)

        # NOTE: sequence_dtype is non-None if this spec is for an annotation
        # like `List[int]` or `tuple`. For `List[int]`, self.dtype is int, and
        # self.sequence_dtype is list.

        # enum_values is a list of the parameter's allowed values
        self.enum_values: List = self.options.get('enum', [])

        # min and max numerical values
        self.min_value: Any[int, float] = self.options.get('min', -1.0)
        self.max_value: Any[int, float] = self.options.get('max', 1.0)

        # select the proper "fit" function based on the parameter's datatype.
        # The fit function converts a raw float, generated by a genetic
        # algorithm, into a corresponding argument value.
        if self.dtype is int:
            self._fit_func = self._fit_int
            # assume we generally want positive int values, like list indices,
            # so set min and max different from default
            if self.min_value is None:
                self.min_value = 0
            if self.max_value is None:
                self.max_value = 1000
        elif self.dtype is str:
            self._fit_func = self._fit_string
            self.max_value = None
            self.min_value = None
        elif self.dtype is bool:
            self._fit_func = self._fit_bool
        else:
            self._fit_func = self._fit_float

        # if the parameter is non-scalar (e.g. is a list, set, tuple),
        # self.size defines its desired length.
        if self.sequence_dtype is not None:
            self.size = self.options.get('size')
            if self.size is None or self.size < 1:
                raise ValueError(f'{param} options must specify a size')
        else:
            self.size = 1

    @property
    def name(self) -> str:
        return self.param.name

    def fit(self, x: float) -> object:
        """
        Convert a raw floating point value, generated by a genetic algorithm,
        into a corresponding parameter value to apply to the owner function
        (which we are optimizing).
        """
        return self._fit_func(x)  # delegate based on the self.dtype

    def _fit_float(self, x: float) -> float:
        """
        Return valid float argument from raw genetic algorithm output value.
        """
        if self.min_value is not None:
            x = max(self.min_value, x)
        if self.max_value is not None:
            x = min(self.max_value, x)
        return x

    def _fit_int(self, x: float) -> int:
        """
        Return valid int argument from raw genetic algorithm output value.
        """
        if self.min_value is not None:
            x = max(self.min_value, x)
        if self.max_value is not None:
            x = min(self.max_value, x)
        return int(x)

    def _fit_string(self, x: float) -> str:
        """
        Return valid str argument from raw genetic algorithm output value.
        """
        values = self.enum_values
        return values[int(abs(x)) % len(values)]

    def _fit_bool(self, x: float) -> bool:
        """
        Return valid bool argument from raw genetic algorithm output value.
        """
        return x >= 0

    def _infer_type(self, param: Parameter) -> Tuple[Type, Optional[Type]]:
        """
        Givent an inspect.Parameter's annotation, resolve and return
        corresponding type object(s). If the annotation is a scalar type,
        like an int or str, we return just (int, None); however, if the
        annotation is something like `List[int]`, we'd return (int, list). By
        default, we return (float, None) -- that is, a single float dtype.
        """
        scalar_types = (int, str, float, bool)
        default_dtype = float
        ann = param.annotation

        if not ann:
            # use defaults
            return (default_dtype, None)

        if isinstance(ann, type):
            if issubclass(ann, scalar_types):
                # the parameter is an int, float, str, etc.
                return (ann, None)
            elif issubclass(ann, (tuple, list, set)):
                # we have a sequence but non-specified inner type,
                # so we use the default (float)
                return (default_dtype, ann)
        elif ann is Text:
            # Text is an alias for str
            return (ann, None)
        elif ann in (Tuple, List, Set):
            # extract inner type from non-scalar annotation, like List[int]
            dtype_map = {Tuple: tuple, List: list, Set: set}
            seq_dtype = dtype_map[ann]
            if ann.__args__:
                if len(ann.__args__) > 1:
                    raise ValueError(
                        f'cannot autotune sequences of anything'
                        f'but floats and ints but got {param}'
                    )
                inner_dtype  = ann.__args__[0]
                if inner_dtype in scalar_types or inner_dtype is Text:
                    return (inner_dtype, seq_dtype)
            else:
                # use defaults
                return (default_dtype, seq_dtype)

        raise ValueError(f'cannot autotune {param} parameter')
