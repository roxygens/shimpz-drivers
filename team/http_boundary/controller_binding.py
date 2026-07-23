"""Import-time binding for hosted controller domain modules."""

from types import ModuleType

_controller: ModuleType | None = None


def bind(controller: ModuleType) -> None:
    global _controller
    _controller = controller


def current() -> ModuleType:
    if _controller is None:
        raise RuntimeError("hosted controller domains were imported before binding")
    return _controller
