from problog.engine_stack import NODE_FALSE


class SimpleProbabilisticBuiltIn(object):
    """Simple builtin that does cannot be involved in a cycle or require engine information and has 0 or more results."""

    def __init__(self, base_function):
        self.base_function = base_function

    def __call__(self, *args, **kwdargs):
        callback = kwdargs.get("callback")
        results = self.base_function(*args, **kwdargs)
        output = []
        if results:
            for i, result in enumerate(results):
                if not result[1] is NODE_FALSE:
                    output += callback.notifyResult(
                        kwdargs["engine"].create_context(result[0], parent=result[0]),
                        result[1],
                        i == len(results) - 1,
                    )
            if output:
                return True, output
            else:
                return True, callback.notifyComplete()
        else:
            return True, callback.notifyComplete()

    def __str__(self):  # pragma: no cover
        return str(self.base_function)
