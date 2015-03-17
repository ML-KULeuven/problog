#!/usr/bin/env python3
# encoding: utf-8
"""
SMILE (.xdsl) Bayesian network format convertor for ProbLog.

Version: 0.4
Author: Tom Sydney Kerckhove

Notes:
    - You can change the delimiter patterns at the top of this file.
    - Nodes with two states are considered boolean, but they are considered
      multi-valued (for the purpose of naming) if its states dont have
      recognizable implicit names (true/false etc).
    - You can change the list of all recognized implicit boolean values at
      the top of this file.
"""
from __future__ import print_function

import re
import sys
import argparse
import xml.etree.ElementTree as ET

desc = "SMILE (.xdls) Bayesian network format convertor for ProbLog (.pl)."

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("input_file", type=str, help="Input .xdsl file")
parser.add_argument("-o", "--output_file", type=str, help="ProbLog file")
parser.add_argument("-0", "--remove_all_zero_probabilities", action="store_true", help="Removes any mention of a probability if it zero.")
args = parser.parse_args()

### Delimiters
BETWEEN_FACTS_AND_PROBS = "::"
END_OF_LINE = "."
DECLARATION = " :- "
CONJUNCTION = ", " 
DISJUNCTION = "; "
NEGATION = "\+"

# Definition: a cool boolean node is a boolean node that has states in the COOL_BOOLEANS list.
COOL_BOOLEANS = [
                    ['t','f'],
                    ['true','false'],
                    ['True','False'],
                    ['yes','no'],
                    ['Yes','No'],
                    ['y','n'],
                    ['pos','neg'],
                    ['aye','nay']
                ]

# The cartesian product of a list of lists.
# This gives you a list of tuples
def cartesian (lists):
    if lists == []: return [()]
    return [x + (y,) for x in cartesian(lists[:-1]) for y in lists[-1]] 


# A representation of a bayesian node in a network
class Node(object):
    def __init__(self, xmlNode):
        
        # Id of node
        self.id = xmlNode.get('id')
        self.id = self.id.lower()

        # Possible states of node
        self.states = []
        for state in xmlNode.findall('state'):
            self.states.append(state.get('id'))

        # All the _names_ of parent nodes.
        parentsNode = xmlNode.find('parents')
        if parentsNode is not None:
            self.parents = re.split(' ', parentsNode.text)
        else:
            self.parents = []
        
        probabilitiesNode = xmlNode.find('probabilities')
        if probabilitiesNode is not None:
            self.probabilities = list(map(float,re.split(' ', probabilitiesNode.text)))
        else:
            self.probabilities = []

        # If this is a cool boolean node, find the better statelist to use
        if len(self.states) == 2:
            good_boolean = []
            for stateList in COOL_BOOLEANS:
                if sorted(stateList) == sorted(self.states): # This is the one
                    good_boolean = stateList
            self.cool_boolean_state_list = good_boolean
        
    def has_parents(self):
        return len(self.parents) > 0
    
    def is_cool_boolean(self):
        return len(self.states) == 2 and self.cool_boolean_state_list

    def state_name(self, state):
        if self.is_cool_boolean():
            if state == self.cool_boolean_state_list[0]:
                return self.id
            else:
                return NEGATION + self.id
        else:
            return self.id + "_" + state

    def toProblog(self, allNodes):
        
        # wrapper to find in dictionary
        def find(name):
            return allNodes[name.lower()]

        # Find all parent nodes and all states that need to be considered per node.
        allStates = []
        parentNodes = []
        for name in self.parents:
            parent = find(name)
            parentNodes.append(parent)
            allStates.append(parent.states)
        allStates.append(self.states)

        # Take the cartesian product of all states. The result is every possible combination of states that need to be considered.
        cartesian_product = cartesian(allStates)

        # List of tuples (disjunctions, conjunctions)
        lines = []

        # Current line
        currentDisjunctions = []
        currentConjunctions = []

        # Amount of own states.
        n = len(self.states)

        # We're going through the table of probabilities in *column-major* order
        # and build up a list of (dis,con) tuples to represent a line later
        for i in range(len(self.probabilities)):
            row = i % len(self.states)
            probability = self.probabilities[i]
            state_name = self.state_name(self.states[row])
            if self.is_cool_boolean() and state_name.startswith(NEGATION):
                continue
            else:
                currentDisjunctions.append((probability,state_name))
            
            # if we're done with disjunctions
            if row == len(self.states)-1 or self.is_cool_boolean():
                # If we need conjunctions
                if self.has_parents():
                    stateTuple = cartesian_product[i]
                    tupleLen = len(self.parents)
                    for j in range(tupleLen):
                        parent = parentNodes[j]
                        name = parent.state_name(stateTuple[j])#result += parent.state_name(stateTuple[j])
                        currentConjunctions.append(name)
                        if j == tupleLen - 1: # At end of conjunctions
                            lines.append((currentDisjunctions, currentConjunctions))
                            currentDisjunctions = []
                            currentConjunctions = []
                else: # No need for conjunctions, end here
                    lines.append((currentDisjunctions, []))
                    currentDisjunctions = []
                    currentConjunctions = []

        # Now we transform the tuple (dis,con) per line into a line of problog code.
        result = ""
        for (dis,con) in lines:
            disStrs = []
            for (prob,name) in dis:
                if prob == 0.0 and args.remove_all_zero_probabilities: # A probability of 0.0 can be ommited.
                    continue
                disStrs.append(str(prob) + BETWEEN_FACTS_AND_PROBS + name)
            if not disStrs: # If no disjunctions (because 0.0 gets filtered), don't output this line.
                continue
            result += DISJUNCTION.join(disStrs)
            if con: # any conjunctions
                result += DECLARATION
                result += CONJUNCTION.join(con)
            result += END_OF_LINE
            result += "\n"

        return result

def xdsl_to_pl(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    nodeNodes = root.find("nodes")

    nodes = {}
    allCptXMLNodes = nodeNodes.findall('cpt')
    for child in allCptXMLNodes:
        n = Node(child)
        nodes[n.id] = n
    
    lines = []
    for id in nodes:
        lines.append(nodes[id].toProblog(nodes))

    return sorted(lines)


if __name__ == "__main__":
    lines = xdsl_to_pl(args.input_file)

    output_file = sys.stdout
    if args.output_file:
        output_file = open(args.output_file,'w')

    for line in lines:
        print(line, file=output_file)

    output_file.close()
