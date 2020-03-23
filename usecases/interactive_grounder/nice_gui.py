from __future__ import print_function

import sys

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

# from PyQt4.QtCore import *


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


class MainWindow(QtGui.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        treeview = TreeDraw()

        model = QtGui.QTextEdit()
        program = QtGui.QTextEdit()
        button = QtGui.QPushButton("Ground")
        button.setMinimumHeight(50)

        hbox = QtGui.QHBoxLayout()

        vbox = QtGui.QVBoxLayout()
        # vbox.addStretch(1)
        vbox.addWidget(model)
        vbox.addWidget(button)
        vbox.addWidget(program)
        model.setMaximumWidth(400)
        program.setMaximumWidth(400)

        hbox.addWidget(treeview)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

        self.setGeometry(50, 50, 1024, 768)
        self.setWindowTitle("Interactive Grounder")
        self.show()


class TreeDraw(QtGui.QWidget):

    NODE_WIDTH = 240
    NODE_HEIGHT = 30

    def __init__(self):
        super(TreeDraw, self).__init__()

        self.tree = {}

    def paintEvent(self, event):
        p = QtGui.QPainter()
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

            sy = float(maxy - miny) / (self.height())
            sx = float(maxx - minx) / (self.width())

            self.coordinates = []
            self.nodes = []

            for node, coord in coordnodes:
                x, y = coord
                if sx > 0:
                    x = (x - minx) / sx
                if sy > 0:
                    y = (y - miny) / sy

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

        color = QtGui.QColor(255, 255, 255)
        if node["name"].startswith("message"):
            color = QtGui.QColor(255, 255, 0)
        elif node.get("cycle"):
            color = QtGui.QColor(255, 0, 0)

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


def main(**kwdargs):
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('inputfile')
    return parser


if __name__ == "__main__":
    main(**vars(argparser().parse_args()))
