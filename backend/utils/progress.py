from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable, Iterator
import queue as _q

_ocr_queue: _q.Queue = _q.Queue()
_sink: ContextVar[Callable[[str], None] | None] = ContextVar("progress_sink", default=None)


def post(msg: str) -> None:
    sink = _sink.get()
    if sink is not None:
        sink(msg)
    else:
        _ocr_queue.put(msg)


def get_queue() -> _q.Queue:
    return _ocr_queue


@contextmanager
def sink(fn: Callable[[str], None]) -> Iterator[None]:
    token = _sink.set(fn)
    try:
        yield
    finally:
        _sink.reset(token)