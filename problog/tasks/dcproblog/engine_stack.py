from problog.engine_stack import StackBasedEngine, EvalNode


class EvalAnd(EvalNode):
    def __init__(self, **parent_args):
        EvalNode.__init__(self, **parent_args)
        self.to_complete = 1
        self.engine.stats[3] += 1

    def __call__(self):
        return False, [self.createCall(self.node.children[0], identifier=None)]

    def new_result(self, result, node=0, source=None, is_last=False):
        if source is None:  # Result from the first conjunct.
            # We will create a second conjunct, which needs to send a 'complete' signal.
            self.to_complete += 1
            if is_last:
                # Notify self that this conjunct is complete. ('all_complete' will always be False)
                all_complete, complete_actions = self.complete()
                # if False and node == NODE_TRUE :
                #     # TODO THERE IS A BUG HERE
                #     # If there is only one node to complete (the new second conjunct) then
                #     #  we can clean up this node, but then we would lose the ground node of
                #     #  the first conjunct.
                #     # This is ok when it is deterministically true.  TODO make this always ok!
                #     # We can redirect the second conjunct to our parent.
                #     return (self.to_complete==1),
                # [ self.createCall( self.node.children[1], context=result, parent=self.parent ) ]
                # else :
                return (
                    False,
                    [
                        self.createCall(
                            self.node.children[1], context=result, identifier=node
                        )
                    ],
                )
            else:
                # Not the last result: default behaviour
                return (
                    False,
                    [
                        self.createCall(
                            self.node.children[1], context=result, identifier=node
                        )
                    ],
                )
        else:  # Result from the second node
            # Make a ground node
            # print(source, node)
            # This has changed to normal problog
            ############
            if self.target.is_density(node):
                self.target.density_node_body[node] = source
                target_node = 0
            else:
                target_node = self.target.add_and((source, node), name=None)
            ###########
            if is_last:
                # Notify self of child completion
                all_complete, complete_actions = self.complete()
            else:
                all_complete, complete_actions = False, []
            if all_complete:
                return True, self.notifyResult(result, target_node, is_last=True)
            else:
                return False, self.notifyResult(result, target_node, is_last=False)

    def complete(self, source=None):
        self.to_complete -= 1
        if self.to_complete == 0:
            return True, self.notifyComplete()
        else:
            assert self.to_complete > 0
            return False, []

    def node_str(self):  # pragma: no cover
        return ""

    def __str__(self):  # pragma: no cover
        return EvalNode.__str__(self) + " tc: %s" % self.to_complete


class StackBasedEngineHAL(StackBasedEngine):
    def __init__(self, label_all=False, **kwdargs):
        StackBasedEngine.__init__(self, label_all=label_all, **kwdargs)

    def eval_conj(self, **kwdargs):
        return self.eval_default(EvalAnd, **kwdargs)
