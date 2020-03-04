****************************
The ProbLog grounding engine
****************************

Preliminaries
=============

The ProbLog grounding engine tranforms a ProbLog model into a propositional logic formula.

Database format
===============

The input of the ProbLog grounder is a database of clauses.
Such a database consists of a list of nodes.


Enabling arbitrary execution order
==================================

The ProbLog grounding engine by default follows Prolog's SLD-resolution evaluation order.
It is however also capable of evaluating choice-points in arbitrary order.

In order to enable this feature, you should create a new subclass of StackBasedEngine.

.. code-block:: python

    from problog.engine_stack import StackBasedEngine

    class RandomOrderEngine(StackBasedEngine):

        def __init__(self):
            StackBasedEngine.__init__(self, unbuffered=True)

        def init_message_stack(self):
            return RandomOrderQueue(self)

    class RandomOrderQueue(MessageAnyOrder):

        def __init__(self, engine):
            MessageAnyOrder.__init__(self, engine)
            # Keep two queues: one for 'result' and 'complete' messages, one for 'eval' messages.
            self.messages_rc = []
            self.messages_e = []

        def append(self, message):
            if message[0] == 'e':
                self.messages_e.append(message)
            else:
                self.messages_rc.append(message)

        def pop(self):
            if self.messages_rc:
                # Process 'result' and 'complete' messages first (keep them in order)
                msg = self.messages_rc.pop(-1)
                return msg
            else:
                # Pick a random 'eval' message.
                i = random.randint(0, len(self.messages_e)-1)
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