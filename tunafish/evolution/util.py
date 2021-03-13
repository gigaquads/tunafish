from typing import Dict

from inspect import Parameter, Signature


def is_parameter_required(parameter: Parameter) -> bool:
    return parameter.kind in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.POSITIONAL_ONLY
    )


def get_required_parameters(signature: Signature) -> Dict:
    return {
        k: v for k, v in signature.parameters.items()
        if is_parameter_required(v)
    }


def clamp(x, lower=0.0, upper=1.0):
    # XXX: Deprecated. Use np.clip instead
    return min(max(lower, x), upper)