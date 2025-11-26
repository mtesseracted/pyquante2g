# pyquante22/arb_math/__init__.py

# from .core import *
# from .eri import *
# from .functions import *

# from . import core as _core

# __all__ = [name for name, obj in vars(_core).items() if callable(obj)]

# Optional: define __all__ to control what gets imported with `from arb_math import *`
# __all__ = []
# __all__ += core.__all__
# __all__ += eri.__all__
# __all__ += functions.__all__

# __init__.py

if True: # this version is needed for dynamic updates to global constants
    from . import core as _core
    from . import functions as _funcs
    # from . import eri as _eri

    # helper: all public callables in a module
    def _public_callables(mod):
        return {
            name
            for name, obj in vars(mod).items()
            if not name.startswith("_") and callable(obj)
        }

    _CORE_CALLABLES  = _public_callables(_core)
    _FUNCS_CALLABLES = _public_callables(_funcs)
    # _ERI_CALLABLES = _public_callables(_eri)

    # non-callable constants we still want to expose
    _EXTRA_CONSTANTS = {"pi", "eps", "azero", "aone", "ahalf"}

    __all__ = sorted(_CORE_CALLABLES | _FUNCS_CALLABLES | _EXTRA_CONSTANTS)


    def __getattr__(name):
        # constants & callables from arb_math
        if name in _CORE_CALLABLES or name in _EXTRA_CONSTANTS:
            return getattr(_core, name)

        # callables from functions.py
        if name in _FUNCS_CALLABLES:
            return getattr(_funcs, name)

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


    def __dir__():
        return sorted(set(list(globals().keys()) + list(__all__)))
