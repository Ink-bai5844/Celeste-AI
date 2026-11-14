"""Fast shallow evolution of data-only frozen dataclass simulation states.

Unlike dataclasses.replace this deliberately supports only ordinary __dict__
dataclasses without constructor/post-init work. Each branch owns a new field
dictionary; immutable tuple/frozenset values are shared exactly as replace does.
"""
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import TypeVar

T = TypeVar("T")


@lru_cache(maxsize=16)
def _field_names(cls: type) -> frozenset[str]:
    if not is_dataclass(cls) or not cls.__dataclass_params__.frozen:
        raise TypeError("evolve requires a frozen dataclass")
    if hasattr(cls, "__post_init__"):
        raise TypeError("evolve cannot bypass dataclass post-init work")
    if "__dict__" not in dir(cls):
        raise TypeError("evolve requires a dataclass with an instance dictionary")
    declared = fields(cls)
    if any(not field.init for field in declared):
        raise TypeError("evolve requires constructor fields only")
    return frozenset(field.name for field in declared)


def evolve(state: T, /, **changes) -> T:
    """Clone a data-only frozen state, rejecting misspelled/unknown fields."""
    cls = type(state)
    allowed = _field_names(cls)
    if not changes.keys() <= allowed:
        unknown = ", ".join(sorted(changes.keys() - allowed))
        raise TypeError(f"Unknown {cls.__name__} field(s): {unknown}")
    result = object.__new__(cls)
    result.__dict__.update(state.__dict__)
    result.__dict__.update(changes)
    return result
