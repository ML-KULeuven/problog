class State(dict):
    # TODO make immutable

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        """Update state by replacing data for keys in other.

        :param other: dictionary with values
        :return: new state
        """
        res = State()
        for sk, sv in res.items():
            res[sk] = sv
        for ok, ov in other.items():
            res[ok] = ov
        return res

    def __mult__(self, other):
        """Update state by discarding current state and replacing it with other.

        :param other: dictionary with values
        :return: new state
        """
        res = State()
        for ok, ov in other.items():
            res[ok] = ov
        return res

    def __or__(self, other):
        """Update state by combining values.

        :param other: dictionary with values
        :return: new state
        """
        res = State()
        for sk, sv in self.items():
            res[sk] = sv
        for ok, ov in other.items():
            if ok in res:
                if isinstance(type(ov), set):
                    res[ok] = sv | ov
                else:
                    res[ok] = sv + ov
            else:
                res[ok] = ov
        return res

    def __hash__(self):
        return hash(tuple([(k, tuple(v)) for k, v in self.items()]))


class Context(list):

    def __init__(self, parent, state=None):
        list.__init__(self, parent)
        if state is None:
            self.state = get_state(parent)
        else:
            self.state = state
        if self.state is None:
            self.state = State()

    def __repr__(self):
        return '%s {%s}' % (list.__repr__(self), self.state)


class FixedContext(tuple):

    def __new__(cls, parent):
        n = tuple.__new__(cls, parent)
        n.state = get_state(parent)
        return n

    def __repr__(self):
        return '%s {%s}' % (tuple.__repr__(self), self.state)

    def __hash__(self):
        return tuple.__hash__(self) + hash(self.state)

    def __eq__(self, other):
        return tuple.__eq__(self, other) and self.state == get_state(other)


def get_state(c):
    if hasattr(c, 'state'):
        return c.state
    else:
        return None