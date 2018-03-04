from __future__ import division
from collections import Counter
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Node:
    def __init__(self, data):
        """
            self.data - training_data for this node
            self.majority_class - the class to predict if this is a leaf Node
            self.split_attr - the attribute used to make the branch
            self.split_value - the value of the attribute used for splitting
            self.left - Node object trained on all data having split_attr <= split_value
            self.right - Node object trained on all data having split_attr > split_value
        """
        self.data = data
        self.majority_class = None
        self.split_attr = None
        self.split_value = None
        self.left = None
        self.right = None

    def compute_entropy(self, classes):
        """
            !!! Given a list(distribution) of classes, return the entropy.
            Ex: I/p - [1,1,1]
                O/p - 0
        """
        entropy = 0.0
        for classLabel in set(classes):
            temp = 0.0
            for i in classes:
                if i == classLabel:
                    temp += 1
            entropy -= temp / len(classes) * math.log(temp / len(classes), 2)
        return entropy

        pass

    def get_values_and_classes(self, attr):
        """
        *** Fill in this documentation string
        :param attr: index of attribute, in this case "0,1,2,3".
        :return: a list of tuples, each tuple contains the value of given attribute index and corresponding class label.
        """
        return [(self.data[ind][attr], self.data[ind].class_) for ind in range(len(self.data))]

    def get_left_split(self, attr, value):
        """
        *** Fill in this documentation string
        :param attr: index of attribute, in this case "0,1,2,3".
        :param value: given value, which is the threshold for splitting the data
        :return: a list of datum instances, each datum instance is at the left of the threshold,
         in other words the value at the given attribute is less than or equal threshold(value).
        """
        return [datum for datum in self.data if datum[attr] <= value]

    def get_right_split(self, attr, value):
        """
        *** Fill in this documentation string
        :param attr: index of attribute, in this case "0,1,2,3".
        :param value: given value, which is the threshold for splitting the data
        :return:a list of datum instances, each datum instance is at the right of the threshold,
         in other words the value at the given attribute is large than the threshold(value).
        """
        return [datum for datum in self.data if datum[attr] > value]

    def get_data_for_left(self):
        """
        *** Fill in this documentation string
        :return: a list of datum instances whose split_attr <= split_value,
        which is split by the best attribute and threshold computed by compute_best_split
        """
        return [datum for datum in self.data if datum[self.split_attr] <= self.split_value]

    def get_data_for_right(self):
        """
        *** Fill in this documentation string
        :return: a list of datum instances whose split_attr > split_value
        which is split by the best attribute and threshold computed by compute_best_split
        """
        return [datum for datum in self.data if datum[self.split_attr] > self.split_value]

    def is_pure(self):
        """
        *** Returns True/False; checks if the node has all samples belonging to the same class.
        :return:
        """
        return len(set([x.class_ for x in self.data])) == 1

    def get_class(self, datum):
        """
        This method passes the datum through the nodes of the tree
        recursively till the leaf node and returns the class.
        :param datum: object of the Datum class
        :return: label of the datum
        """
        # !!! Write code.
        current_root = self
        while True:
            attris = datum.features
            if current_root.majority_class is not None:
                return current_root.majority_class
            else:
                if attris[current_root.split_attr] <= current_root.split_value:
                    current_root = current_root.left
                else:
                    current_root = current_root.right



        pass

    def compute_best_split(self):
        attrs_entropies = {ind: {"gain": 0, "value": 0} for ind in range(len(self.data[0].features))}
        current_entropy = self.compute_entropy([datum.class_ for datum in self.data])
        for ind in range(len(self.data[0].features)):
            values_and_classes = self.get_values_and_classes(ind)
            values = sorted([x[0] for x in values_and_classes])
            best_value = None
            best_gain = 0
            for prev_value, next_value in zip(values, values[1:]):
                value = (prev_value + next_value) / 2
                # !!! Using the above get_* methods, write code to compute gain for each value of the attribute.

                left_Entropy = self.compute_entropy([datum.class_ for datum in self.get_left_split(ind, value)])
                right_Entropy = self.compute_entropy([datum.class_ for datum in self.get_right_split(ind, value)])
                numOfLeftData = float(len(self.get_left_split(ind, value)))
                numOfRightData = float(len(self.get_right_split(ind, value)))
                numOfTotalData = float(len(self.data))
                sum_Entropy = (numOfLeftData / numOfTotalData) * left_Entropy + (
                            numOfRightData / numOfTotalData) * right_Entropy
                gain = current_entropy - sum_Entropy

                if gain > best_gain:
                    best_gain = gain
                    best_value = value
            attrs_entropies[ind]["gain"] = best_gain
            attrs_entropies[ind]["value"] = best_value

        self.split_attr = max(attrs_entropies, key=lambda x: attrs_entropies[x]["gain"])
        self.split_value = attrs_entropies[self.split_attr]["value"]


class Datum:
    def __init__(self, features, class_):
        self.features = features
        self.class_ = class_

    def __getitem__(self, item):
        # *** Read about operator overloading
        return self.features[item]


class DecisionTree:
    def __init__(self, train_data, test_data):
        self.root = Node(train_data)
        self.test_data = test_data

    def train(self):
        """
        Code to train the decision tree recursively.
        :return:
        """
        stack_nodes = [self.root]
        # !!! Write code to train decision tree. If the node is pure, set the majority_class attribute.
        # Use .pop(0) to pop the top of the stack

        while len(stack_nodes) != 0:
            if len(stack_nodes) == 0:
                continue
            root = stack_nodes.pop(0)
            root.compute_best_split()
            if root.is_pure():
                root.majority_class = root.data[0].class_
                continue
            else:
                root.left = Node(root.get_data_for_left())
                root.right = Node(root.get_data_for_right())
                stack_nodes.append(root.left)
                stack_nodes.append(root.right)
        pass

    def predict(self, datum):
        return self.root.get_class(datum)

    def test_accuracy(self):
        pos = 0
        for datum in self.test_data:
            pred_class = self.predict(datum)
            if pred_class == datum.class_:
                pos += 1
        return pos / len(self.test_data)


if __name__ == "__main__":
    np.random.seed(5)
    iris_data = np.loadtxt('iris.data', delimiter=',')
    np.random.shuffle(iris_data)

    features = iris_data[:, 0:4]
    targets = iris_data[:, 4]
    data_size = int(0.8 * features.shape[0])

    train_feats, train_classes = features[:data_size].tolist(), targets[:data_size].tolist()
    test_feats, test_classes = features[data_size:].tolist(), targets[data_size:].tolist()

    train_data = [Datum(feats, class_) for feats, class_ in zip(train_feats, train_classes)]
    test_data = [Datum(feats, class_) for feats, class_ in zip(test_feats, test_classes)]

    dt = DecisionTree(train_data, test_data)
    dt.train()
    print(dt.test_accuracy())

    dt = DecisionTreeClassifier(criterion="entropy")
    dt.fit(train_feats, train_classes)
    print(np.mean(dt.predict(test_feats) == test_classes))
