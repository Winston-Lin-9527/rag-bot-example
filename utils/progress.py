import queue as _q

_ocr_queue: _q.Queue = _q.Queue()


def post(msg: str) -> None:
    _ocr_queue.put(msg)


def get_queue() -> _q.Queue:
    return _ocr_queue