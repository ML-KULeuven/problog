Cycles
======


Unlike Prolog, ProbLog's evaluation is buffered, that is, all solutions to a call are first collected before returning them to the caller.
However, when a cycle is found, this strategy is abandoned and the engine proceeds unbuffered within the cycle.

Let us consider the following non-trivial cyclic program and the graphical representation of its structure.

.. code-block:: prolog

	a :- b.		a :- d.		a :- c.	
	b :- f.		
	f :- a.		f :- c.
	c :- e.		c :- b.


.. graphviz::

	digraph Cycle {
		a2 [label="a'"];
		b2 [label="b'"];
		c2 [label="c'"];
		a -> b;
		b -> a2;
		b -> f;
		f -> c;
		c -> e;
		c -> b2;
		a -> c2;
		a -> d;
		cd [style=invis];
		cd -> c2 [style=invis];
		cd -> d [style=invis];
	}

This program could be part of a larger program.

We call the node ``a`` the *cycle root*.
Node ``b`` is a *subcycle root* of ``a``, and ``c`` is a *subcycle* root of ``b``.
The nodes ``a'``, ``b'`` and ``c'`` are *cycle children* of nodes ``a``, ``b`` and ``c`` respectively. 
Each (sub)cycle root contains references to the roots of its subcycles and its children.
The nodes on the path between a cycle root and a cycle child are *cyclic* nodes (for example ``f``).
The other nodes (e.g. ``d`` and ``e``) are regular nodes.
(For simplicity we shall store all information about subcycles in the cycle root node.)

The engine keeps track of active calls, that is, calls that are currently being evaluated.
During regular operation, we can detect a cycle when a call is made that is already active.
When this happens we perform the following actions:
	
 * The original call node is marked as the cycle root.
 * Register cycle root with engine.
 * The new call node is marked as a cycle child.
 * The new call node is registered with the original call node as a cycle child.
 * The new call queues a ``createCycle(call, child, False)`` message to its parent.
 * The new call queues a ``closeCycle(True)`` message to be send to the cycle root.

When the engine is already processing cycles, it is possible that other cycles occur.
In the example above, we have a cycle on ``b`` inside a cycle on ``a``.
Note that it is possible that the cycle on ``b`` is discovered first.

When a cycle is detected while there is already a cycle root, we follow a slightly different procedure.

 * Check whether the original node is an ancestor of the current root (comparing pointers should suffice).
   		If it is, it will become the new cycle root:
			* Copy subcycle information from the old root to the new root
			* Remove cycle root marker
			* Register the new root with the engine
			* Register the old root as a subcycle of the new root
			* The new call queues a ``closeCycle(True)`` message to be send to the new cycle root.
		If it isn't:
			* Register the original as subcycle of root node.
 * The new call node is marked as a cycle child.
 * The new call node is registered with the original call node as a cycle child.
 * The new call queues a ``createCycle(call, child, False)`` message to its parent.


The role of a node affects the way it processes message.
  
**Regular define nodes**

 * ``newResult``: buffers the result
 * ``complete``: sends all results and a complete signal
 * ``createCycle``: become a cyclic node, flush buffer to parents, forward message to parents
 * ``closeCycle``: should not happen
 
**Cycle root**

 * ``newResult``: buffers the result to parents, send directly to children
 * ``complete``: unregister cycle root from engine, flush all results to parents, send complete signal to parents
 * ``createCycle``: do nothing (discard message)
 * ``closeCycle(True)``: forward message to subcycles (with argument set to False), send ``complete`` to cycle children (order shouldn't matter)

**Subcycle root**
 
 * ``newResult``: forward result to parents and children
 * ``complete``: forward to parents
 * ``createCycle``: forward message to parents 
 * ``closeCycle(True)``: ignore (this happens when this node used to be the cycle root)
 * ``closeCycle(False)``: forward message to subcycles, send ``complete`` to cycle children (order shouldn't matter)
 
**Cyclic node**

 * ``newResult``: forward result to parents
 * ``complete``: forward to parents
 * ``createCycle``: forward message to parents (could discard?)
 * ``closeCycle``: should not happen
 
**Not node**

 * ``newResult``: buffer results
 * ``complete``: send (negated) results to parents
 * ``createCycle``: raise exception NegativeCycle
 * ``closeCycle``: should not happen

**Other**

 * not affected by cycles
 
The correctness of this approach depends on the observation that sending a ``complete`` signal can never generate a new result.
This observation is only valid if there are no buffered nodes (such as ``Not``).	
