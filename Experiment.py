# * Experiment.py
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

import pandas as pd
import matplotlib.pyplot as plt

from ClassifierAlgorithm import (
    ClassifierAlgorithm,
    simplekNNClassifier,
    kdTreeKNNClassifier,
)


class Experiment:
    """It is a Base Class to construct and there is no subclass for now."""

    def __init__(self, dataset, labels, lst_classifier):
        """The Base class Experiment constructor, it only has simpleKNNClassifier for now
        dataset: Dataframe
        labels: List
        lst_classifier: List

        return: None
        """
        print("The Experiment Constructor is build.")
        self.dataset = dataset
        self.labels = labels
        self.lst_classifier = lst_classifier
        for classifier in lst_classifier:
            if classifier != "simpleKNNClassifier":
                raise ValueError("Classifier not available")

    def runCrossVal(self, k):
        """The runCrossVal function, k will be the number of folders to run the cross validation
        k: int

        return: all labels
        """

        print("The runCrossVal is build.")
        # initial the k folders
        self.k = k
        # get the average rows
        avg_rows = self.dataset.shape[0] // k
        # place to save all labels
        all_predicted_labels = []
        # run each classifier
        for i in range(len(self.lst_classifier)):
            # save predicted labels
            predicted_labels = []
            # for each k, run simple KNN Classifier
            for j in range(k):
                classifier = simplekNNClassifier()
                # check if is simple KNN Classifier
                if self.lst_classifier[i] == "simpleKNNClassifier":
                    # split the testData and test label
                    if j == (k - 1):
                        testData = self.dataset.iloc[j * avg_rows :]
                        test_labels = self.labels.iloc[j * avg_rows :]
                    else:
                        testData = self.dataset.iloc[j * avg_rows : (j + 1) * avg_rows]
                        test_labels = self.labels.iloc[
                            j * avg_rows : (j + 1) * avg_rows
                        ]
                    # get the train data and train label
                    trainingData = self.dataset.drop(testData.index)
                    train_labels = self.labels.drop(test_labels.index)
                    # do classifier on train set
                    classifier.train(trainingData, train_labels)
                    # combine all predicted labels
                    predicted_labels.extend(classifier.test(testData, 5))
                else:
                    raise ValueError("kdTreeKNNClassifier not implemented yet.")
            # return all predicted labels
            all_predicted_labels.append(predicted_labels)
        return all_predicted_labels

    def score(self):
        """The score function, get the classifier name and accuracy of each classifier
        and return as a table

        return: None
        T(n): O(n^3)
        S(n): O(n^2)
        T(n) is for each classifier, I will run the classifier.test which has O(n^2), it will become (n^2)*n = n^3
        S(n) is for each classfier, I will save all the data, which has O(n^2), since I reuse the space, it will not
        go for O(n^3)
        """
        # initial output of this function
        print("The score is build.")  # step:1   space:0
        # title of the table
        print("Classifier name      | Accuracy")  # step:1   space:0
        # check the classifier
        for i in range(len(self.lst_classifier)):  # step:n   space:1
            # get correct number of labels
            correct = 0  # step:n   space:1
            # if the classifier is simple KNN Classifier
            if self.lst_classifier[i] == "simpleKNNClassifier":  # step:3n   space:1
                # modify the dataset for calculate
                # initial the simple KNN Classifier
                classifier = simplekNNClassifier()  # step:n   space:1
                # combine the dataset for random split
                whole_data = pd.concat(
                    [pd.DataFrame(self.labels), self.dataset], axis=1
                )  # step:n^2   space:n^2
                # random split
                trainingData = whole_data.sample(
                    frac=0.8, random_state=200
                )  # step:n   space:n^2
                # save the test Data for test
                testData = whole_data.drop(trainingData.index)  # step:n   space:n^2
                # save the train set label
                train_true_label = trainingData.iloc[:, 0]  # step:n   space:n^2
                # save the true test label for further comparing
                test_label = testData.iloc[:, 0].to_list()  # step:n   space:n
                # get rid of the train label of train data
                trainingData = trainingData.iloc[:, 1:]  # step:n   space:n^2
                # get rid of the test label of test data
                testData = testData.iloc[:, 1:]  # step:n   space:n^2
                # do the classifier on train data
                classifier.train(trainingData, train_true_label)  # step:n   space:n^2
                # run test on test Data
                predicted_test_label = classifier.test(
                    testData, 5
                )  # step:(n^2)*n = n^3   space:n
                # get the correct labels compare to incorrect labels
                for j in range(len(predicted_test_label)):  # step:n^2   space:1
                    if predicted_test_label[j] == test_label[j]:  # step:n^2   space:0
                        correct += 1  # step:n^2?   space:0
                # count the accuracy of correct labels
                accuracy = correct / len(predicted_test_label)  # step:2n   space:1
                # output the table
                print(f"{self.lst_classifier[i]}  | {accuracy}")  # step:n   space:0
            # if it is kdtreeKNNClassifier, not builted
            else:
                raise ValueError("kdTreeKNNClassifier is not builted yet.")

    def confusionMatrix(self, k):
        """The confusion function, compute and display a confusion matrix for each classifier
        and shown and a table

        return: None
        T(n): O(n^3)
        S(n): O(n^2)
        T(n) is for each classifier, I will run the classifier.test which has O(n^2), it will become (n^2)*n = n^3
        S(n) is for each classfier, I will save all the data, which has O(n^2), since I reuse the space, it will not
        go for O(n^3)
        """
        # initial output of the function
        print("The confusion Matrix is build.")  # step:1   space:0
        # initial output of the table
        print("Confusion Matrix:")  # step:1   space:0
        # check the classifier
        for i in range(len(self.lst_classifier)):  # step:n   space:1
            # if it is a simple KNN Classifier
            if self.lst_classifier[i] == "simpleKNNClassifier":  # step:n   space:1
                # organized the data for further calculation
                # initial the simple KNN Classifier
                classifier = simplekNNClassifier()  # step:n   space:1
                # combine the data frame afor random split
                whole_data = pd.concat(
                    [pd.DataFrame(self.labels), self.dataset], axis=1
                )  # step:n^2   space:n^2
                # get the labels
                whole_data_label = whole_data.iloc[:, 0]  # step:n   space:n
                # split the sample randomly
                trainingData = whole_data.sample(
                    frac=0.8, random_state=200
                )  # step:n   space:n^2
                # save the test data
                testData = whole_data.drop(trainingData.index)  # step:n   space:n^2
                # save the train data label
                train_true_label = trainingData.iloc[:, 0]  # step:n   space:n
                # save the test data label
                test_label = testData.iloc[:, 0].to_list()  # step:n   space:n
                # get rid of the training label's training set
                trainingData = trainingData.iloc[:, 1:]  # step:n   space:n^2
                # get rid of the test label's test set
                testData = testData.iloc[:, 1:]  # step:n   space:n^2
                # train the label
                classifier.train(trainingData, train_true_label)  # step:n   space:2n
                # predicted all test label
                predicted_test_label = classifier.test(
                    testData, k
                )  # step:(n^2)*n = n^3   space:0
                # save all the category names
                label_names = whole_data_label.value_counts().keys()  # step:n   space:n
                # count each of the category names
                predicted_test_label_counts = [0] * (
                    len(label_names) ** 2
                )  # step:n   space:n
                # split predicted labels into nine groups since the labels has three types
                for j in range(len(predicted_test_label)):  # step:n^2   space:1
                    if test_label[j] == label_names[0]:  # step:n^2   space:1
                        # True, predict correct in first group
                        if (
                            predicted_test_label[j] == label_names[0]
                        ):  # step:n^2   space:1
                            predicted_test_label_counts[0] += 1  # step:n^2   space:0
                        # False, predicted on second label but actually in first label
                        elif (
                            predicted_test_label[j] == label_names[1]
                        ):  # step:n^2   space:1
                            predicted_test_label_counts[1] += 1  # step:n^2   space:0
                        # False, predicted on third label but actually in first label
                        else:
                            predicted_test_label_counts[2] += 1  # step:n^2   space:1
                    elif test_label[j] == label_names[1]:  # step:n^2   space:0
                        # False, predicted on first label but actually in second label
                        if (
                            predicted_test_label[j] == label_names[0]
                        ):  # step:n^2   space:1
                            predicted_test_label_counts[3] += 1  # step:n^2   space:0
                        # True, predict correct in second group
                        elif (
                            predicted_test_label[j] == label_names[1]
                        ):  # step:n^2   space:1
                            predicted_test_label_counts[4] += 1  # step:n^2   space:0
                        # False, predicted on third label but actually in second label
                        else:
                            predicted_test_label_counts[5] += 1  # step:n^2   space:0
                    else:
                        # False, predicted on first label but actually in third label
                        if (
                            predicted_test_label[j] == label_names[0]
                        ):  # step:n^2   space:1
                            predicted_test_label_counts[6] += 1  # step:n^2   space:0
                        # False, predicted on second label but actually in third label
                        elif (
                            predicted_test_label[j] == label_names[1]
                        ):  # step:n^2   space:1
                            predicted_test_label_counts[7] += 1  # step:n^2   space:0
                        # True, predict correct in third group
                        else:
                            predicted_test_label_counts[8] += 1  # step:n^2   space:0
                # print out the confusion matrix
                # print top labels
                print(
                    f"Actual                                   |"  # step:n   space:0
                    f"{label_names[0]}|"
                    f"{label_names[1]}|"
                    f"{label_names[2]}"
                )
                # print first row of first prediction
                print(
                    f"{label_names[0]}      |"  # step:n   space:0
                    f"                {predicted_test_label_counts[0]}                 |"
                    f"                {predicted_test_label_counts[1]}                |"
                    f"                     {predicted_test_label_counts[2]}"
                )
                # print second row of second prediction
                print(
                    f"{label_names[1]}        | "  # step:n   space:0
                    f"                {predicted_test_label_counts[3]}                 |"
                    f"                {predicted_test_label_counts[4]}               |"
                    f"                     {predicted_test_label_counts[5]}"
                )
                # print third row of third prediction
                print(
                    f"{label_names[2]}|"  # step:n   space:0
                    f"                {predicted_test_label_counts[6]}                 |"
                    f"                {predicted_test_label_counts[7]}                |"
                    f"                     {predicted_test_label_counts[8]}"
                )
            # if it is the kd Tree Classifier, it is not builted yet
            else:
                raise ValueError(
                    "kdTreeKNNClassifier is not builted yet."
                )  # step:n   space:0

        self.tpr_a = predicted_test_label_counts[0] / (  # step:7   space:1
            predicted_test_label_counts[0]
            + predicted_test_label_counts[1]
            + predicted_test_label_counts[2]
        )
        self.fpr_a = (  # step：11   space:1
            predicted_test_label_counts[3] + predicted_test_label_counts[6]
        ) / (
            predicted_test_label_counts[3]
            + predicted_test_label_counts[4]
            + predicted_test_label_counts[5]
            + predicted_test_label_counts[6]
            + predicted_test_label_counts[7]
            + predicted_test_label_counts[8]
        )
        self.tpr_g = predicted_test_label_counts[4] / (  # step:7   space:1
            predicted_test_label_counts[3]
            + predicted_test_label_counts[4]
            + predicted_test_label_counts[5]
        )
        self.fpr_g = (  # step:11   space:1
            predicted_test_label_counts[1] + predicted_test_label_counts[7]
        ) / (
            predicted_test_label_counts[0]
            + predicted_test_label_counts[1]
            + predicted_test_label_counts[2]
            + predicted_test_label_counts[6]
            + predicted_test_label_counts[7]
            + predicted_test_label_counts[8]
        )
        self.tpr_c = predicted_test_label_counts[8] / (  # step:7   space:1
            predicted_test_label_counts[6]
            + predicted_test_label_counts[7]
            + predicted_test_label_counts[8]
        )
        self.fpr_c = (  # step:11   space:1
            predicted_test_label_counts[2] + predicted_test_label_counts[5]
        ) / (
            predicted_test_label_counts[0]
            + predicted_test_label_counts[1]
            + predicted_test_label_counts[2]
            + predicted_test_label_counts[3]
            + predicted_test_label_counts[4]
            + predicted_test_label_counts[5]
        )

    def ROC(self, tpr, fpr):
        """Plot the ROC graph based on true positive rate and false positive rate

        tpr: List[List[float]]
        fpr: List[List[float]]
        return: None

        step:55 + O(n^4) (from test) + 6n + 21(from ROC)
        space:6 (from confusion matrix) + O(n^3) (from test) + n^2 (from ROC)
        to conclude: T(n): O(n^4)   S(n): O(n^3)
        """
        plt.figure()  # step:1   space:n^n
        plt.plot(fpr[0], tpr[0], color="b", label="Adelie Penguin")  # step:2n   space:0
        plt.plot(fpr[1], tpr[1], color="g", label="Gentoo Penguin")  # step:2n   space:0
        plt.plot(
            fpr[2], tpr[2], color="r", label="Chiinstrap Penguin"
        )  # step:2n   space:0
        plt.plot(
            [0, 1], [0, 1], color="navy", linestyle="--", label="ROC 0.5 line"
        )  # step:7   space:0
        plt.xlim([0.0, 1.0])  # step:1   space:0
        plt.ylim([0.0, 1.0])  # step:7   space:0
        plt.xlabel("False Positive Rate")  # step:1   space:0
        plt.ylabel("True Positive Rate")  # step:1   space:0
        plt.title("ROC graph")  # step:1   space:0
        plt.legend(loc="lower right")  # step:1   space:0
        plt.show()  # step:1   space:0
