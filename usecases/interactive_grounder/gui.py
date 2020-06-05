#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

from __future__ import print_function

import sys
import igraph
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from threading import Condition, Thread

from problog.formula import LogicFormula
from problog.program import PrologFile
from problog.logic import Term

import problog.engine_stack as es
from problog.engine import DefaultEngine
from problog.logic import Term, term2str


def tree2nodes(tree, n=0):
    edges = []
    result = [tree]
    if "children" in tree:
        for c in tree["children"]:
            childnodes, childedges = tree2nodes(c, n + len(result))
            edges.append((n, n + len(result)))
            edges += childedges
            result += childnodes
    return result, edges


def tree2graph(tree):

    if tree:
        nodes, edges = tree2nodes(tree)
        if edges:
            g = igraph.Graph(edges)
            layout = g.layout_reingold_tilford(root=[0])
            return zip(nodes, layout.coords), edges
        else:
            return [], []
    else:
        return [], []


class TreeDraw(QWidget):

    NODE_WIDTH = 240
    NODE_HEIGHT = 30

    def __init__(self):
        super(TreeDraw, self).__init__()

        self.initUI()
        self.tree = {}

    def initUI(self):
        self.show()

    def paintEvent(self, event):
        p = QPainter()
        p.begin(self)
        self.drawTree(p)
        p.end()

    def drawTree(self, p):
        tree = self.tree
        coordnodes, coordedges = tree2graph(self.tree)

        if coordnodes:

            minx = min([c[0] for n, c in coordnodes])
            maxx = max([c[0] for n, c in coordnodes])
            miny = min([c[1] for n, c in coordnodes])
            maxy = max([c[1] for n, c in coordnodes])

            sy = float(maxy - miny) / (self.height() - 200)
            sx = float(maxx - minx) / (self.width() - 200)

            self.coordinates = []
            self.nodes = []

            for node, coord in coordnodes:
                x, y = coord
                if sx > 0:
                    x = 20 + (x - minx) / sx
                if sy > 0:
                    y = 20 + (y - miny) / sy

                self.coordinates.append((x, y))
                self.nodes.append(node)
                self.drawNode(p, (x, y), node)

            for a, b in coordedges:
                x1, y1 = self.coordinates[a]
                x2, y2 = self.coordinates[b]
                p.drawLine(
                    x1 + self.NODE_WIDTH / 2,
                    y1 + self.NODE_HEIGHT,
                    x2 + self.NODE_WIDTH / 2,
                    y2,
                )

    def drawNode(self, p, coord, node):
        x, y = coord

        color = QColor(255, 255, 255)
        if node["name"].startswith("message"):
            color = QColor(255, 255, 0)
        elif node.get("cycle"):
            color = QColor(255, 0, 0)

        p.fillRect(x, y, self.NODE_WIDTH, self.NODE_HEIGHT, color)
        p.drawRect(x, y, self.NODE_WIDTH, self.NODE_HEIGHT)
        p.drawText(
            x,
            y,
            self.NODE_WIDTH,
            self.NODE_HEIGHT,
            Qt.AlignVCenter | Qt.AlignLeft,
            node["text"],
        )

    def mousePressEvent(self, event):
        print("clicked", event.x(), event.y())

        print("############## ENGINE STATE ##################")
        message_order.engine.printStack()
        print(message_order)
        print("##############################################")

        x = event.x()
        y = event.y()
        for i, c in enumerate(self.coordinates):
            if (
                c[0] <= x <= c[0] + self.NODE_WIDTH
                and c[1] <= y <= c[1] + self.NODE_HEIGHT
            ):
                name = self.nodes[i]["name"]
                if name.startswith("message_"):
                    index = int(name[8:])
                    message_order.make_choice(index)


# Create an PyQT4 application object.
a = QApplication(sys.argv)

tree = {}

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TreeDraw()

w.tree = tree

# Set window size.
w.resize(1024, 768)

# Set window title
w.setWindowTitle("Interactive Grounder")


sysout = sys.stdout


class MessageOrderInteractive(es.MessageOrder1):
    def __init__(self, engine):
        es.MessageOrder1.__init__(self, engine)
        self.cv = Condition()
        self.choice = None

    def make_choice(self, identifier):
        self.cv.acquire()
        print("make choice", identifier, file=sysout)
        self.choice = identifier
        self.cv.notify()
        self.cv.release()

    def get_choice(self):
        self.cv.acquire()
        print("############# GROUND PROGRAM #################")
        print(target.to_prolog())
        print("##############################################")

        print("############## ENGINE STATE ##################")
        self.engine.printStack()
        print(self)
        print("##############################################")

        while self.choice is None:
            self.cv.wait()
            print("get choice", self.choice, file=sysout)
        result = self.choice
        self.choice = None
        self.cv.release()
        return result

    def append(self, message):

        if message[0] == "e":
            print(
                "append",
                message[0:3],
                message[3].get("call"),
                message[3].get("context"),
                file=sysout,
            )
            self.messages_e.append(message)
            self.update_tree()
        else:
            self.messages_rc.append(message)

    def pop(self):
        if self.messages_rc:
            msg = self.messages_rc.pop(-1)
            self.update_tree()
            return msg
        else:
            i = self.get_choice()
            res = self.messages_e.pop(i)
            self.update_tree()
            return res

    def update_tree(self):
        w.tree = self.stack2tree()
        w.repaint()

    def node2text(self, node):
        if type(node).__name__ == "EvalDefine":
            call = node.call
            return term2str(Term(call[0], *call[1]))
        else:
            res = self.dbnode2text(node.database, node.node_id, node.node)
            if res:
                return res
            return str(node)

    def dbnode2text(self, database, dbnode, node, context=None):
        if type(node).__name__ == "clause":
            return term2str(database._extract(node.child))
        elif type(node).__name__ in ("call", "conj", "fact"):
            n = database._extract(dbnode)
            # if context:
            #     n = n.apply(context)
            return term2str(n)

    def message2text(self, i, message):
        #
        msgtype, dbnode, args, context = message
        node = context["database"].get_node(dbnode)
        # print (context['call'], node, context['context'])

        res = self.dbnode2text(context["database"], dbnode, node, context["context"])
        if res:
            return res

        call = message[3]["call"]
        return "%s: %s" % (i, term2str(Term(call[0], *call[1])))

    def stack2tree(self):
        nodes = {}

        for i, node in enumerate(self.engine.stack):
            if node is not None:
                name = "stack_%s" % i
                if node.parent is not None:
                    parent = "stack_%s" % node.parent
                else:
                    parent = None
                cc = hasattr(node, "is_cycle_child") and node.is_cycle_child
                nodes[name] = {
                    "name": name,
                    "text": self.node2text(node),
                    "parent": parent,
                    "cycle": cc,
                }
                if parent is not None:
                    if not "children" in nodes[parent]:
                        nodes[parent]["children"] = []
                    nodes[parent]["children"].append(nodes[name])

        for i, message in enumerate(self.messages_e):
            parent = message[3]["parent"]
            text = self.message2text(i, message)
            name = "message_%i" % i
            nodes[name] = {"name": name, "text": text, "parent": message[1]}
            parent = "stack_%s" % message[3]["parent"]
            if not "children" in nodes[parent]:
                nodes[parent]["children"] = []
            nodes[parent]["children"].append(nodes[name])

        if nodes:
            return nodes["stack_0"]
        else:
            return None


message_order = None


class InteractiveEngine(es.StackBasedEngine):
    def __init__(self, **kwdargs):
        es.StackBasedEngine.__init__(self, unbuffered=True, label_all=True, **kwdargs)

    def init_message_stack(self):
        global message_order
        if message_order is None:
            message_order = MessageOrderInteractive(self)
        return message_order


def ground(model, query, target):
    eng = InteractiveEngine()
    db = eng.prepare(model)
    print(db)
    target = eng.ground(db, query, target=target, name=(False, query, "query"))
    return target


filename = sys.argv[1]
model = PrologFile(filename)
eng = DefaultEngine()

queries = eng.query(model, Term("query", None))

query = queries[0][0]

target = LogicFormula()

t = Thread(target=ground, args=(model, query, target))
t.daemon = True
t.start()

# Show window
w.show()

sys.exit(a.exec_())
