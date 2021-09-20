import builtins

old_print = builtins.print


def flushed_print(*args, **kwargs):
    kwargs['flush'] = True
    return old_print(*args, **kwargs)


builtins.print = flushed_print
