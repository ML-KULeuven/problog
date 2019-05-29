import random


class MessageQueue(object):
    """A queue of messages."""

    def __init__(self):
        pass

    def append(self, message):
        """Add a message to the queue.

        :param message:
        :return:
        """
        raise NotImplementedError('Abstract method')

    def __iadd__(self, messages):
        """Add a list of message to the queue.

        :param messages:
        :return:
        """
        for message in messages:
            self.append(message)
        return self

    def cycle_exhausted(self):
        """Check whether there are messages inside the cycle.

        :return:
        """
        raise NotImplementedError('Abstract method')

    def pop(self):
        """Pop a message from the queue.

        :return:
        """
        raise NotImplementedError('Abstract method')

    def __nonzero__(self):
        raise NotImplementedError('Abstract method')

    def __bool__(self):
        raise NotImplementedError('Abstract method')

    def __len__(self):
        raise NotImplementedError('Abstract method')

    def repr_message(self, msg):
        if msg[0] == 'c':
            return 'c(%s)' % msg[1]
        elif msg[0] == 'r':
            return 'r(%s, %s)' % (msg[1], msg[2])
        elif msg[0] == 'e':
            return 'e(%s, %s, %s, %s)' % (msg[1], msg[3].get('call'), msg[3].get('context'), msg[3].get('parent'))

    def __iter__(self):
        raise NotImplementedError('Abstract method')

    def __repr__(self):
        return '[%s]' % ', '.join(map(self.repr_message, self))


class MessageFIFO(MessageQueue):

    def __init__(self, engine):
        MessageQueue.__init__(self)
        self.engine = engine
        self.messages = []

    def append(self, message):
        self.messages.append(message)
        # Inform the debugger.
        if self.engine.debugger:
            self.engine.debugger.process_message(*message)

    def pop(self):
        return self.messages.pop(-1)

    def peek(self):
        return self.messages[-1]

    def cycle_exhausted(self):
        if self.engine.cycle_root is None:
            return False
        else:
            last_message = self.peek()
            return last_message[0] == 'e' and \
                   last_message[3]['parent'] < self.engine.cycle_root.pointer

    def __nonzero__(self):
        return bool(self.messages)

    def __bool__(self):
        return bool(self.messages)

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)


class MessageAnyOrder(MessageQueue):

    def __init__(self, engine):
        MessageQueue.__init__(self)
        self.engine = engine

    def _msg_parent(self, message):
        if message[0] == 'e':
            return message[3]['parent']
        else:
            return message[1]

    def cycle_exhausted(self):
        if self.engine.cycle_root is None:
            return False
        else:
            for message in self:  # TODO cache
                parent = self._msg_parent(message)
                if self.engine.in_cycle(parent):
                    return False
            return True


class MessageOrderD(MessageAnyOrder):

    def __init__(self, engine):
        MessageAnyOrder.__init__(self, engine)
        self.messages = []

    def append(self, message):
        self.messages.append(message)

    def pop(self):
        return self.messages.pop(-1)

    def __nonzero__(self):
        return bool(self.messages)

    def __bool__(self):
        return bool(self.messages)

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)


class MessageOrderDrc(MessageAnyOrder):
    def __init__(self, engine):
        MessageAnyOrder.__init__(self, engine)
        self.messages_rc = []
        self.messages_e = []

    def append(self, message):
        if message[0] == 'e':
            self.messages_e.append(message)
        else:
            self.messages_rc.append(message)

    def pop(self):
        if self.messages_rc:
            msg = self.messages_rc.pop(-1)
            return msg
        else:
            res = self.messages_e.pop(-1)
            return res

    def __nonzero__(self):
        return bool(self.messages_e) or bool(self.messages_rc)

    def __bool__(self):
        return bool(self.messages_e) or bool(self.messages_rc)

    def __len__(self):
        return len(self.messages_e) + len(self.messages_rc)

    def __iter__(self):
        return iter(self.messages_e + self.messages_rc)


class MessageOrder1(MessageAnyOrder):

    def __init__(self, engine):
        MessageAnyOrder.__init__(self, engine)
        self.messages_rc = []
        self.messages_e = []

    def append(self, message):
        if message[0] == 'e':
            self.messages_e.append(message)
        else:
            self.messages_rc.append(message)

    def pop(self):
        if self.messages_rc:
            msg = self.messages_rc.pop(-1)
            # print ('M', msg)
            return msg
        else:
            i = random.randint(0, len(self.messages_e) - 1)
            # print ('MESSAGE', [m[0:2] + (m[3]['context'],) for m in self.messages_e], self.messages_e[i][0:3])
            res = self.messages_e.pop(i)
            return res

    def __nonzero__(self):
        return bool(self.messages_e) or bool(self.messages_rc)

    def __bool__(self):
        return bool(self.messages_e) or bool(self.messages_rc)

    def __len__(self):
        return len(self.messages_e) + len(self.messages_rc)

    def __iter__(self):
        return iter(self.messages_e + self.messages_rc)