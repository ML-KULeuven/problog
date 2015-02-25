LogicFormula
============

.. currentmodule:: problog.formula

.. autoclass:: LogicFormula

Managing logic structure
------------------------

The :class:`LogicFormula` offers several methods for updating the logical structure of the And-Or graph.
The default behavior is to create immutable nodes, which allows the data structure to perform several optimizations.

Each of these methods returns the key of the node that is created.
All atoms in the formula are probabilistic, that is, deterministic nodes are usually optimized away.

There are two special keys that indicate deterministic nodes.

.. autoattribute:: LogicFormula.TRUE
    
    The key of the node indicating deterministically True.

.. autoattribute:: LogicFormula.FALSE
    
    The key of the node indicating deterministically False.

These do typically not occur in the data structure itself.

.. automethod:: LogicFormula.addAtom
.. automethod:: LogicFormula.addNot
.. automethod:: LogicFormula.addAnd
.. automethod:: LogicFormula.addOr


.. automethod:: LogicFormula.addDisjunct

.. automethod:: LogicFormula.iterNodes
.. automethod:: LogicFormula.getNode

.. automethod:: LogicFormula.isTrue
.. automethod:: LogicFormula.isFalse
.. automethod:: LogicFormula.isProbabilistic

Managing node names
-------------------

.. automethod:: LogicFormula.addName

.. automethod:: LogicFormula.addQuery
.. automethod:: LogicFormula.addEvidence

.. automethod:: LogicFormula.getNames
.. automethod:: LogicFormula.getNamesWithLabel
.. automethod:: LogicFormula.getNodeByName

.. automethod:: LogicFormula.queries
.. automethod:: LogicFormula.evidence
.. automethod:: LogicFormula.named

.. autoattribute:: LogicFormula.LABEL_QUERY
.. autoattribute:: LogicFormula.LABEL_EVIDENCE_POS
.. autoattribute:: LogicFormula.LABEL_EVIDENCE_NEG
.. autoattribute:: LogicFormula.LABEL_EVIDENCE_MAYBE
.. autoattribute:: LogicFormula.LABEL_NAMED

Managing node weights
---------------------

Atoms in a :class:`LogicFormula` can have a weight.
These weight are simply stored in the data structure without making any assumptions on their significance.
Default weights are set by :func:`addAtom`.

The :class:`LogicFormula` offers the following functions for retrieving the weights.

.. automethod:: LogicFormula.getWeights
.. automethod:: LogicFormula.extractWeights


Output
------

.. automethod:: LogicFormula.__str__
.. automethod:: LogicFormula.toDot
.. automethod:: LogicFormula.toProlog
