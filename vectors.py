def growing_index(xs, step):
    # will include start and end points
    xs = [x for x in xs]
    res = [0]
    base = xs[0]
    for i, x in enumerate(xs):
        accum = x - base
        if accum >= step:
            res.append(i)
            accum = 0
            base = x
    return res


def midpoint(xs, p):
    return p * (xs[-1] - xs[0]) + xs[0]


def qs(n, start=False, end=False):
    step = 1 / n
    gen = range(0 if start else 1, n + (1 if end else 0))
    return [step * i for i in gen]


def percentile_index(xs, ps):
    xs = [x for x in xs]
    res = []
    ps = ps if isinstance(ps, list) else [ps]
    p_stream = iter(ps)
    p = next(p_stream)
    v = midpoint(xs, p)
    for i, x in enumerate(xs):
        if x >= v:
            res.append(i)
            try:
                p = next(p_stream)
                v = midpoint(xs, p)
            except StopIteration:
                break
            continue
    return res
