#!/usr/bin/env python3
# encoding: utf-8
"""
HUGIN (.net) Bayesian network format convertor for ProbLog.

Author: Michiel Derhaeg
"""
from __future__ import print_function

import sys
import re
import argparse

desc = "HUGIN (.net) Bayesian network format convertor for ProbLog (.pl)."

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("input_file", type=str, help="Input .net file")
parser.add_argument("-o", "--output_file", type=str, help="ProbLog file")
args = parser.parse_args()

netregex = re.compile("net[\s\n]*{(\n|.)*}")
noderegex = re.compile("node\s+([_\w-]+)[\n\s]*{([^}]*)}")
potentialregex = re.compile("potential\s*\(\s*([_\w-]+)\s*(\|\s*([_\w-]+\s*)+)?\)[\n\s]*{([^}]*)}")
statementRegex = re.compile("([_\w-]+)[]\s\n]*=[\n\s]*([^;]+);")
wordRegex = re.compile("(\w+)")
GoodBoolStates = [["true","false"],["yes","no"],["y","n"], ["t","f"]]

def netlog(inputfilepath):
    inputfile = open(inputfilepath, "r")
    netcode = re.sub("%.*","",inputfile.read()).lower()
    NodeList = []
    PotentialList =  []
    for nodematch in noderegex.finditer(netcode):
        NodeList.append(parseNode(nodematch.group(1),nodematch.group(2)))
    for potentialmatch in potentialregex.finditer(netcode):
        PotentialList.append(parsePotential(potentialmatch.groups()))
    return makeProblog(NodeList,PotentialList)

def parseNode(name,body):
    newnode = Node(name)
    for statementMatch in statementRegex.finditer(body):
        parseStatement(newnode, statementMatch.group(1), statementMatch.group(2))
    return newnode

def parseStatement(node, element, value):
    if element == "states":
        for match in wordRegex.finditer(value):
            node.states.append(match.group(1))

def parsePotential(groups):
    node = groups[0]
    othernodes = groups[1]
    body = groups[3]
    newpotential = Potential(node)
    if othernodes:
        for nodematch in re.finditer("([_\w-]+)",othernodes):
            newpotential.othernodes.append(nodematch.group(0))
    for statementMatch in statementRegex.finditer(body):
        parseDataStatement(newpotential,statementMatch.group(1),statementMatch.group(2))
    return newpotential

def parseDataStatement(potential,element,value):
    if element == "data":
        value = re.sub("\)\s+\(",")(",re.sub("[\n\r]","",value ))
        data = []
        value = re.sub("[)(]"," ",value)
        output = ""
        for char in value:
            if char == " ":
                if len(output):
                    data.append(output)
                output = ""
            else:
                output += char
        potential.data = data

def normalizeData(data,nrOfStates):
    if (len(data)):
        for j in range(0,len(data), nrOfStates):
            sumofdata = 0
            for i in range(j, nrOfStates+j):
                sumofdata += float(data[i])
            for i in range(j, nrOfStates+j):
                data[i] = float(data[i]) / sumofdata
        data = list(map(str,data))
    return data

class Node():
    def __init__(self,_name):
        self.name = _name
        self.states = []
    def nameWithState(self):
        reverse = list(self.states)
        reverse.reverse()
        if self.states in GoodBoolStates:
            return [self.name, "\+" + self.name]
        else:
            reverse = list(self.states)
            reverse.reverse()
            if reverse in GoodBoolStates:
                return ["\+" + self.name, self.name]
            else:
                names = list(self.states)
                for i in range(0,len(names)):
                    names[i] = self.name + "_" + names[i]
                return names

class Potential():
    def __init__(self,node):
        self.node = node
        self.othernodes = []
        self.data = []
    def dimension(self):
        return 1 + len(self.othernodes)

def makeProblog(nodes,potentials):
    output = ""
    for p in potentials:
        mainNode = findnode(p.node,nodes)
        p.data = normalizeData(p.data, len(mainNode.states))
        if not(len(p.data)): #TODO fix
            continue
        if p.dimension() == 1:
            if (len(mainNode.states) == 2):
                output += p.data[0] + "::" + mainNode.nameWithState()[0] + ".\n"
            else:
                for i in range(0,len(p.data)):
                    output += p.data[i] + "::" + mainNode.nameWithState()[i] + ".\n"
        else:
            cartlist = []
            for n in p.othernodes:
                node = findnode(n,nodes)
                cartlist.append(node.nameWithState())
            cart = cartesian(cartlist)
            for i in range(0,len(p.data), len(mainNode.states)):
                chances = {}
                if (len(mainNode.states) == 2):
                    chances[0] = p.data[i] + "::" + mainNode.nameWithState()[0]
                else:
                    for j in range(0,len(mainNode.states)):
                        chances[j] = p.data[i+j] + "::" + mainNode.nameWithState()[j]
                output += ";".join(chances.values()) + " :- "
                aboutFacts = {}
                for l in range(0,len(cart[int(i/len(mainNode.states))])):
                    aboutFacts[l] = cart[int(i/len(mainNode.states))][l]
                output += ",".join(aboutFacts.values())
                output += ".\n"

    return output

def findnode(name,nodes):
    output = None
    for n in nodes:
        if n.name == name:
            output = n
            break
    return output

def cartesian (lists):
    if lists == []: return [()]
    return [x + (y,) for x in cartesian(lists[:-1]) for y in lists[-1]]

if __name__ == "__main__":
    output_file = sys.stdout
    if args.output_file:
        output_file = open(args.output_file,'w')

    output_file.write(netlog(args.input_file))
    output_file.close()

