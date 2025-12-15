# * ClassifierAlgorithm.py
# *
# * ANLY 555 <2022 Spring>
# * Project <5. Advanced Topics>
# *
# * Due on: <May 4th 2022>
# * Author(s): <Ke Shi>
# *
# *
# * In accordance with the class policies and Georgetown's
# * Honor Code, I certify that, with the exception of the
# * class resources and those items noted below, I have neither
# * given nor received any assistance on this project other than
# * the TAs, professor, textbook and teammates.
# *

import math
from statistics import mode
import pandas as pd
import numpy as np


class Tree:
    """It is a class called Tree

    It will be a healper class for kdKNNtreeClassifier
    """

    def __init__(self):
        """It is the constructor of Tree"""
        self.value = None
        self.label = None
        self.location = None
        self.depth = None
        self.leftChild = None
        self.rightChild = None


class ClassifierAlgorithm:
    """It is a Base Class to construct.

    Two subclass will be inherited, which is
    simplekNNClassifier: for simple KNN classifier
    kdTreeKNNClassifier: for KD TREE KNN clasifier
    """

    def __init__(self):
        """The Base class Classifier Algorithm constructor
        set trainingData, true_label, testData to None, and k = 5

        return: None
        """
        # print("The Classifier Algorithm Constructor is build.")
        self.trainingData = None
        self.true_label = None
        self.testData = None
        self.k = 5

    def train(self, trainingData, true_label):
        """The train function, it will initialize the trainingData, and true_label to
        be further process, it is a abstract base function, will be inherited or
        overwrite in subclass

        return: None
        """
        print("The Classifier Algorithm Train is build")
        self.trainingData = trainingData
        self.true_label = true_label

    def test(self, testData, k=5):
        """The test function, it will initialize the testData, and k to be further process,
        it is a abstract base function, will be inherited or overwrite in subclass

        return: None
        """
        print("The Classifier Algorithm Test is build")
        self.testData = testData
        self.k = k


class simplekNNClassifier(ClassifierAlgorithm):
    """It is a subclass called simpleKNNClassifier
    and is inherited from the base class ClassifierAlgorithm.

    It will be dealing with simple KNN specifically.
    """

    def __init__(self):
        """This is the constructor for simplekNNClassifier,
        it is inherited from the base class ClassifierAlgorithm.
        """
        super().__init__()
        # print("The simple kNN Classifier Constructor is inherit.")

    def train(self, trainingData, true_label):
        """This is a train function for simple KNN.
        It will override the train function from the base class ClassifierAlgorithm.
        trainingData = DataFrame
        true_label = List

        return: None
        """
        # print("The simple kNN Classifier Train is build")
        self.trainingData = trainingData
        self.true_label = true_label

    def test(self, testData, k=5):
        """This is a test function for simple KNN.
        It will override the test function from the base class ClassifierAlgorithm.
        It will do the simple KNN classifier for each row of testData, and get the
        predicted label to return
        testData: DataFrame
        k: num

        return: List[str]
        T(n): O(n^2)
        S(n): O(n)
        T(n) is for the data set, I will calculate the Euclidian Distance for each row of train set, compare the
        results for test data sets rows, one n for test length, one n for train set length
        S(n) is for the data set, I will save the labels, it will be O(n)
        """
        # print("The simple kNN Classifier Test is build")      # step:1   space:1
        self.testData = testData  # step:1   space:n
        self.k = k  # step:1   space:1
        predicted_test_label = []  # step:1   space:1
        # do the Euclidean Distance for each row of testData
        for i in range(len(testData)):  # step:n   space:n
            ED = []  # step:n   space:n
            labels = []  # step:n   space:n
            # predicted the labels based on all trainingData
            for j in range(len(self.trainingData)):  # step:n^2   space:n
                total_distance = (
                    (self.trainingData.iloc[j][0] - testData.iloc[i][0]) ** 2
                    + (self.trainingData.iloc[j][1] - testData.iloc[i][1]) ** 2
                    + (self.trainingData.iloc[j][2] - testData.iloc[i][2]) ** 2
                    + (self.trainingData.iloc[j][3] - testData.iloc[i][3]) ** 2
                )  # step:12n^2   space:1
                total_distance = math.sqrt(total_distance)  # step:n^2   space:1
                ED.append(total_distance)  # step:n^2   space:0
            # get the mode of the closest k labels
            for l in range(k):  # step:n^2   space:n
                labels.append(
                    self.true_label.iloc[ED.index(min(ED))]
                )  # step:4n^2   space:0
                ED.pop(ED.index(min(ED)))  # step:2n^2   space:-1
            # save all the test data labels
            predicted_test_label.append(mode(labels))  # step:n   space:n
        return predicted_test_label  # step:1   space:0


class kdTreeKNNClassifier(ClassifierAlgorithm):
    """It is a subclass called kdTreeKNNClassifier
    and is inherited from the base class ClassifierAlgorithm.

    It will be dealing with kd Tree KNN specifically.
    """

    def __init__(self):
        """This is the constructor for kdTreeKNNClassifier,
        it is inherited from the base class ClassifierAlgorithm.
        """
        super().__init__()
        print("The kd Tree Classifier Constructor is inherit.")
        self.num_attributes = 0
        self.trainingData = None
        self.train_true_label = None
        self.testData = None
        self.k = 5
        self.kdtree = None

    def build_tree(self, lst, label, depth):
        """Built tree based on lst, label, and depth and the class Tree()

        lst: list
        label: list
        depth: int
        return: Tree
        T(n): 13 + 3n + nlogn + 2*T(n/2)     -> recursion structure
        T(n) = O(n^2)
        S(n): 8 + 4n + 2*S(n/2)              -> recursion structure
        S(n) = O(nlogn)
        """
        tree = Tree()  # step:1   space:1
        tree.depth = depth  # step:1   space:1
        tree.axis = tree.depth % self.num_attributes  # step:3   space:1
        if len(lst) == 0:  # step:3   space:1
            return None  # step:1   space:1
        values = [x[tree.axis] for x in lst]  # step:n   space:n
        index = np.argsort(values)  # step:nlogn   space:n
        vec_list = lst[index]  # step:n   space:n
        lbls = label[index]  # step:n   space:n
        tree.location = len(vec_list) // 2  # step:2   space:1
        tree.value = vec_list[tree.location]  # step:1   space:1
        tree.label = lbls[tree.location]  # step:1   space:1
        tree.leftChild = self.build_tree(
            vec_list[0 : tree.location], lbls[0 : tree.location], depth + 1
        )  # step:T(n/2)   space:S(n/2)
        tree.rightChild = self.build_tree(
            vec_list[tree.location + 1 :], lbls[tree.location + 1 :], depth + 1
        )  # step:T(n/2)   space:S(n/2)
        return tree

    def train(self, trainingData, train_true_label):
        """This is a train function for kd Tree KNN.
        It will override the train function from the base class ClassifierAlgorithm.

        return: None
        T(n) = 1 + 2n + 2n^2 + O(n^2) (from build_tree function)
        T(n) = O(n^2)
        S(n) = 1 + n + n^2 + O(nlogn) (from build_tree function)
        S(n) = O(n^2)
        Compare to simple KNN, the time complexity looks same, they are both O(n^2), but
        simple KNN will have a more constant time complexity. The Kd-tree time complexity might
        fluctuate a lot, it might be higher than simple KNN, if the tree didn't split the data
        well. But it also might be saving time than KNN.
        For space complexity, the Kd-tree method will be higher than simpleKNN for sure, because
        we need extra space for saving tree structure.
        """
        print("The kd Tree Classifier Train is build")
        self.trainingData = trainingData  # step:n^2   space:n^2
        self.train_true_label = train_true_label  # step:n   space:n
        self.num_attributes = self.trainingData.shape[1]  # step:1   space:1
        self.trainingData = np.array(
            [
                self.trainingData.iloc[i, :].to_numpy()
                for i in range(self.trainingData.shape[0])
            ]
        )  # step:n^2   space:0
        self.train_true_label = self.train_true_label.to_numpy()  # step:n   space:0
        self.kdtree = self.build_tree(
            self.trainingData, self.train_true_label, 0
        )  # step:O(n^2)   space:O(nlogn)  by build_tree function

    def closest(self, value, tree1, tree2):
        """Check which children has smaller distance

        value: float
        tree1: tree
        tree2: tree
        return: tree, float
        T(n) = 6
        T(n) = O(c) -> constant
        S(n) = 5
        S(n) = O(c) -> constant
        """
        dist_to_rt1 = np.linalg.norm(value - tree1.value)  # step:1   space:1
        dist_to_rt2 = np.linalg.norm(value - tree2.value)  # step:1   space:1
        if dist_to_rt1 < dist_to_rt2:  # step:3   space:1
            return tree1, dist_to_rt1  # step:2   space:0
        else:
            return tree2, dist_to_rt2  # step:2   space:2

    def nearest_neighbor(self, tree, lst):
        """check the nearest neightbor of the tree

        tree: tree
        lst: list
        return: tree
        T(n) = 28 + 2*O(c)
        T(n) = O(c)
        S(n) = 4 + 2*O(c)
        S(n) = O(c)
        """
        if tree is None:  # step:2   space:1
            return None  # step:2   space:0
        if lst[tree.axis] < tree.value[tree.axis]:  # step:3   space:1
            next_branch = tree.leftChild  # step:1   space:1
            other_branch = tree.rightChild  # step:1   space:1
        else:
            next_branch = tree.rightChild  # step:1   space:0
            other_branch = tree.leftChild  # step:1   space:0
        temp = self.nearest_neighbor(next_branch, lst)  # step:T(c)   space:S(c)
        if temp is not None:  # step:2   space:1
            best, dist = self.closest(lst, tree, temp)  # step:T(c)   space:S(c)
        else:
            best = tree  # step:1   space:0
            dist = np.linalg.norm(lst - tree.value)  # step:2   space:0
        dist_prime = abs(lst[tree.axis] - tree.value[tree.axis])  # step:4   space:1
        if dist_prime < dist:  # step:3   space:1
            temp = self.nearest_neighbor(other_branch, lst)  # step:T(c)   space:S(c)
            if temp is not None:  # step:2   space:1
                best, temp_dist = self.closest(
                    lst, best, temp
                )  # step:T(c)   space:S(c)
            else:
                best = tree  # step:2   space:0
        return best  # step:2   space:0

    def test(self, testData, k=5):
        """This is a test function for kd Tree KNN.
        It will override the train function from the base class ClassifierAlgorithm.

        return: list[string]
        T(n) = 5 + 2n + 4n^2
        T(n) = O(n^2)
        S(n) = 3 + 5n + n^2
        S(n) = O(n^2)
        Compare to simple KNN, the time complexity looks same, they are both O(n^2), but
        simple KNN will have a more constant time complexity. The Kd-tree time complexity might
        fluctuate a lot, it might be higher than simple KNN, if the tree didn't split the data
        well. But it also might be saving time than KNN.
        For space complexity, the Kd-tree method will be higher than simpleKNN for sure, because
        we need extra space for saving tree structure.
        """
        print("The kd Tree Classifier Test is build")
        if k is not None:  # step:2   space:1
            self.k = k  # step:1   space:1
        self.testData = testData  # step:n^2   space:n^2
        test_shape = self.testData.shape  # step:1   space:1
        test_result = [None] * test_shape[0]  # step:n   space:n
        for i in range(test_shape[0]):  # step:n   space:n
            test_vec = self.testData.iloc[i, :].to_numpy()  # step:n^2   space:n
            neighbor = self.nearest_neighbor(
                self.kdtree, test_vec
            )  # step:n^2   space:n
            test_result[i] = neighbor.label  # step:n^2   space:n
        return test_result  # step:2   space:0
