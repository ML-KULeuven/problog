from sklearn.datasets import load_iris
from sklearn import tree
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import dt2problog
import subprocess as sp
import pickle


def test_iris():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(iris.data, iris.target)
    with open("iris_clf.pkl", "wb") as ofile:
        pickle.dump(clf, ofile)


    with open("iris.dot", 'w') as f:
        tree.export_graphviz(clf, out_file=f,
                             feature_names=iris.feature_names,
                             class_names=iris.target_names,
                             filled=True, rounded=True,
                             special_characters=True)

    rules = dt2problog.Rules(clf, iris.feature_names, iris.target_names)
    print(rules.to_problog(use_comparison=True, min_prob=0.0))
    print("---")
    settings, data = rules.to_probfoil(min_prob=0.0)
    print(settings)
    print(data)
    with open("iris.data", "w") as f:
        print(data, file=f)
    with open("iris.settings", "w") as f:
        print(settings, file=f)
    # Run $ probfoil iris.data iris.settings

test_iris()
