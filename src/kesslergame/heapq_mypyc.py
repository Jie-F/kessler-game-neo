# This is Python standard library's heapq.py, but with comments stripped out, and type hints added so it can be compiled with mypyc to get a fast priority queue!

from typing import TypeVar, Iterable, Callable, Iterator, Optional, Any, Protocol


class SupportsRichComparison(Protocol):
    def __lt__(self: "T", other: "T") -> bool:
        ...

    def __le__(self: "T", other: "T") -> bool:
        ...

    def __gt__(self: "T", other: "T") -> bool:
        ...

    def __ge__(self: "T", other: "T") -> bool:
        ...


T = TypeVar("T", bound=SupportsRichComparison)

__all__ = [
    "heappush", "heappop", "heapify", "heapreplace", "merge",
    "nlargest", "nsmallest", "heappushpop"
]


def heappush(heap: list[T], item: T) -> None:
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)


def heappop(heap: list[T]) -> T:
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt


def heapreplace(heap: list[T], item: T) -> T:
    returnitem = heap[0]
    heap[0] = item
    _siftup(heap, 0)
    return returnitem


def heappushpop(heap: list[T], item: T) -> T:
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0)
    return item


def heapify(x: list[T]) -> None:
    n = len(x)
    for i in reversed(range(n // 2)):
        _siftup(x, i)


def _heappop_max(heap: list[T]) -> T:
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt


def _heapreplace_max(heap: list[T], item: T) -> T:
    returnitem = heap[0]
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem


def _heapify_max(x: list[T]) -> None:
    n = len(x)
    for i in reversed(range(n // 2)):
        _siftup_max(x, i)


def _siftdown(heap: list[T], startpos: int, pos: int) -> None:
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap: list[T], pos: int) -> None:
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


def _siftdown_max(heap: list[T], startpos: int, pos: int) -> None:
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup_max(heap: list[T], pos: int) -> None:
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)


def merge(
    *iterables: Iterable[T],
    key: Optional[Callable[[T], Any]] = None,
    reverse: bool = False,
) -> Iterator[T]:
    h: list[Any] = []
    h_append = h.append

    if reverse:
        _heapify_func = _heapify_max
        _heappop_func = _heappop_max
        _heapreplace_func = _heapreplace_max
        direction = -1
    else:
        _heapify_func = heapify
        _heappop_func = heappop
        _heapreplace_func = heapreplace
        direction = 1

    if key is None:
        for order, it in enumerate(map(iter, iterables)):
            try:
                next_fn = it.__next__
                h_append([next_fn(), order * direction, next_fn])
            except StopIteration:
                pass
        _heapify_func(h)
        while len(h) > 1:
            try:
                while True:
                    value, order, next_fn = s = h[0]
                    yield value
                    s[0] = next_fn()
                    _heapreplace_func(h, s)
            except StopIteration:
                _heappop_func(h)
        if h:
            value, order, next_fn = h[0]
            yield value
            yield from next_fn.__self__
        return

    for order, it in enumerate(map(iter, iterables)):
        try:
            next_fn = it.__next__
            value = next_fn()
            h_append([key(value), order * direction, value, next_fn])
        except StopIteration:
            pass
    _heapify_func(h)
    while len(h) > 1:
        try:
            while True:
                key_value, order, value, next_fn = s = h[0]
                yield value
                value = next_fn()
                s[0] = key(value)
                s[2] = value
                _heapreplace_func(h, s)
        except StopIteration:
            _heappop_func(h)
    if h:
        key_value, order, value, next_fn = h[0]
        yield value
        yield from next_fn.__self__


def nsmallest(n: int, iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> list[T]:
    if n == 0:
        return []

    if key is None:
        # mypy: use list[T] directly
        it = iter(iterable)
        result: list[tuple[T, int]] = [(elem, i) for i, elem in zip(range(n), it)]
        if not result:
            return []
        _heapify_max(result)
        top = result[0][0]
        order = n
        _heapreplace_func = _heapreplace_max
        for elem in it:
            if elem < top:
                _heapreplace_func(result, (elem, order))
                top, _ = result[0]
                order += 1
        result.sort()
        return [elem for (elem, _) in result]

    # Decorate-sort-undecorate
    it2 = iter(iterable)
    result2: list[tuple[Any, int, T]] = [(key(elem), i, elem) for i, elem in zip(range(n), it2)]
    if not result2:
        return []
    _heapify_max(result2)
    top2 = result2[0][0]
    order2 = n
    _heapreplace_func2 = _heapreplace_max
    for elem in it2:
        k = key(elem)
        if k < top2:
            _heapreplace_func2(result2, (k, order2, elem))
            top2, _, _ = result2[0]
            order2 += 1
    result2.sort()
    return [elem for (k, order, elem) in result2]


def nlargest(n: int, iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> list[T]:
    if n == 0:
        return []

    if key is None:
        it = iter(iterable)
        result: list[tuple[T, int]] = [(elem, i) for i, elem in zip(range(0, -n, -1), it)]
        if not result:
            return []
        heapify(result)
        top = result[0][0]
        order = -n
        _heapreplace_func = heapreplace
        for elem in it:
            if top < elem:
                _heapreplace_func(result, (elem, order))
                top, _ = result[0]
                order -= 1
        result.sort(reverse=True)
        return [elem for (elem, _) in result]

    it2 = iter(iterable)
    result2: list[tuple[Any, int, T]] = [(key(elem), i, elem) for i, elem in zip(range(0, -n, -1), it2)]
    if not result2:
        return []
    heapify(result2)
    top2 = result2[0][0]
    order2 = -n
    _heapreplace_func2 = heapreplace
    for elem in it2:
        k = key(elem)
        if top2 < k:
            _heapreplace_func2(result2, (k, order2, elem))
            top2, _, _ = result2[0]
            order2 -= 1
    result2.sort(reverse=True)
    return [elem for (k, order, elem) in result2]
