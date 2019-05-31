
class AbstractMessage(object):
    def __init__(self, target, args, context):
        self._target = target
        self._args = args
        self._context = context

    @property
    def target(self):
        return self._target

    def set_new_target(self, new_target):
        self._target = new_target

    @property
    def args(self):
        return self._args

    @property
    def context(self):
        return self._context

    @property
    def is_eval_message(self):
        return False

    @property
    def is_complete_message(self):
        return False

    @property
    def is_result_message(self):
        return False


class EvalMessage(AbstractMessage):
    def __init__(self, target, args, context):
        super().__init__(target, args, context)

    @property
    def is_eval_message(self):
        return True

    def __str__(self):
        return 'e(%s, %s, %s, %s)' % (
            self.target, self.context.get('call'), self.context.get('context'), self.context.get('parent'))


class ResultMessage(AbstractMessage):
    def __init__(self, target, args, context):
        super().__init__(target, args, context)

    @property
    def is_result_message(self):
        return True

    def __str__(self):
        return 'r(%s, %s)' % (self.target, self.args)


class CompleteMessage(AbstractMessage):
    def __init__(self, target, args, context):
        super().__init__(target, args, context)

    @property
    def is_complete_message(self):
        return True

    def __str__(self):
        return 'c(%s)' % self.target
