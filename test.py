# * test.py
# *
# * ANLY 555 <2022 Spring>
# * Project <4. Transaction Data Set, ROC, and Decision Tree>
# *
# * Due on: <Apr 17th 2022>
# * Author(s): <Ke Shi>
# *
# *
# * In accordance with the class policies and Georgetown's
# * Honor Code, I certify that, with the exception of the
# * class resources and those items noted below, I have neither
# * given nor received any assistance on this project other than
# * the TAs, professor, textbook and teammates.
# *

# =====================================================================
# Testing script for Deliverable 2: Source Code Framework
# =====================================================================

# =====================================================================
# Testing DataSet Class
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# =====================================================================
import pandas as pd
from DataSet import (
    DataSet,
    QuantDataSet,
    QualDataSet,
    TextDataSet,
    TimeSeriesDataSet,
    TransactionDataSet,
    HeterogenousDataSet,
)

time_series_data = "TimeSeriesData/ptbdb_normal.csv"
text_data = "TextData/yelp.csv"
quant_data = "QuantData/Sales_Transactions_Dataset_Weekly.csv"
qual_data = "QualData/multiple_choice_responses.csv"
penguins = "QuantData/penguins_lter.csv"
transaction_data = "TransactionData/groceries.csv"


def DataSetTests():
    print(
        "DataSet Instantiation invokes both the __load() and the\
__readFromCSV() methods."
    )
    data = DataSet(time_series_data)
    print("Check the member methods for DataSet.")
    print("==============================================================")
    print("Now call DataSet.readFromCSV().")
    data.readFromCSV(time_series_data)
    print(data.readFromCSV(time_series_data))
    print("==============================================================")
    print("Now call DataSet.clean().")
    data.clean()
    print("==============================================================")
    print("Now call DataSet.explore().")
    data.explore()
    print("==============================================================")
    print("\n\n")


def TimeSeriesDataSetTests():
    print("In TimeSeries DataSet, check inheritance.")
    data = TimeSeriesDataSet(time_series_data)
    print("===========================================================")
    print("Check member methods.")
    print("Check that clean and explore methods have been override.")
    print("===========================================================")
    print(data.readFromCSV(time_series_data))
    print("TimeSeriesDataSet.clean():")
    data.clean()
    print("===========================================================")
    print("TimeSeriesDataSet.explore():")
    data.explore()
    print("\n\n")


def TextDataSetTests():
    print("In TextDataSet DataSet, check inheritance.")
    data = TextDataSet(text_data)
    print(data)
    print("===========================================================")
    print("Check member methods.")
    print("Check that clean and explore methods have been override.")
    print("===========================================================")
    print("TextDataSet.clean():")
    data.clean()
    print("===========================================================")
    print("TextDataSet.explore():")
    data.explore()
    print("\n\n")


def QuantDataSetTests():
    print("In QuantDataSet, check inheritance.")
    data = QuantDataSet(quant_data)
    print("===========================================================")
    print("Check member methods.")
    print("Check that clean and explore methods have been override.")
    print("===========================================================")
    print("QuantDataSet.clean():")
    data.clean()
    print("===========================================================")
    print("QuantDataSet.explore():")
    data.explore()
    print("\n\n")


def QualDataSetTests():
    print("In QualDataSet, check inheritance.")
    data = QualDataSet(qual_data)
    print("===========================================================")
    print("Check member methods.")
    print("Check that clean and explore methods have been override.")
    print("===========================================================")
    print("QualDataSet.clean():")
    data.clean(method="mode")
    print("===========================================================")
    print("QualDataSet.explore():")
    data.explore()
    print("\n\n\n")


def TransactionDataSetTests():
    print("In Transaction DataSet, check inheritance.")
    data = TransactionDataSet(transaction_data)
    print("===========================================================")
    print("Check member methods.")
    print("Check that clean and explore methods have been override.")
    print("===========================================================")
    print("TransactionDataSet.readFromCSV():")
    data.readFromCSV(transaction_data)
    print("===========================================================")
    print("TransactionDataSet.clean():")
    data.clean()
    print("===========================================================")
    print("TransactionDataSet.explore():")
    data.explore()
    print("===========================================================")
    print("TransactionDataSet.explore(0.23)")
    data.explore(0.23)
    print("===========================================================")
    print("TransactionDataSet.explore(0.1)")
    data.explore(0.1)
    print("\n\n\n")


def HeterogenousDataSetTests():
    print("In Transaction DataSet, check inheritance.")
    data = HeterogenousDataSet([qual_data, quant_data])
    print("===========================================================")
    print("Check member methods.")
    print("Check that clean and explore methods have been override.")
    print("===========================================================")
    print("TransactionDataSet.readFromCSV():")
    data.readFromCSV()
    print("===========================================================")
    print("TransactionDataSet.clean():")
    data.clean()
    print("===========================================================")
    print("TransactionDataSet.explore():")
    data.explore()
    data.select(0)
    print("\n\n\n")


# =====================================================================
# Testing Classifier Class
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# =====================================================================
from ClassifierAlgorithm import (
    ClassifierAlgorithm,
    simplekNNClassifier,
    kdTreeKNNClassifier,
)


def ClassifierAlgorithmTests():
    print("ClassifierAlgorithm Instantiation.")

    print("==============================================================")
    print("Prepare for the dataset Penguin.")
    # readin file
    data = pd.read_csv(penguins)
    # choose columns needed
    data = data.iloc[:, [2, 9, 10, 11, 12]]
    # drop nas
    data = data.dropna()
    # check the datasets
    data["Species"].value_counts()
    # check the datasets
    data.describe()
    # split into train set and test set
    training = data.sample(frac=0.8, random_state=200)
    test = data.drop(training.index)
    # set for training Data without labels
    trainingData = training.iloc[:, 1:6]
    # training data labels
    train_true_label = training.iloc[:, 0]
    # set for testing Data without labels
    testData = test.iloc[:, 1:6]
    # testing Data labels, will not be used in KNN, but will be used in Experiment
    test_true_label = test.iloc[:, 0]
    print("==============================================================")

    classifier = ClassifierAlgorithm()
    print("==============================================================")
    print("Check the member methods for Classifier Algorithm Tests.")
    print("==============================================================")
    print("Now call ClassifierAlgorithm.train().")
    classifier.train(trainingData, train_true_label)
    print("==============================================================")
    print("Now call ClassifierAlgorithm.test().")
    classifier.test(testData, 5)
    print("==============================================================")
    print("\n\n")


def simplekNNClassifierTests():
    print("simpleKNNClassifier Instantiation.")

    print("==============================================================")
    print("Prepare for the dataset Penguin.")
    # readin file
    data = pd.read_csv(penguins)
    # choose columns needed
    data = data.iloc[:, [2, 9, 10, 11, 12]]
    # drop nas
    data = data.dropna()
    # check the datasets
    data["Species"].value_counts()
    # check the datasets
    data.describe()
    # split into train set and test set
    training = data.sample(frac=0.8, random_state=200)
    test = data.drop(training.index)
    # set for training Data without labels
    trainingData = training.iloc[:, 1:6]
    # training data labels
    train_true_label = training.iloc[:, 0]
    # set for testing Data without labels
    testData = test.iloc[:, 1:6]
    # testing Data labels, will not be used in KNN, but will be used in Experiment
    test_true_label = test.iloc[:, 0]
    print("==============================================================")

    classifier = simplekNNClassifier()
    print("===========================================================")
    print("Check member methods.")
    print("Check that train and test methods have been override.")
    print("===========================================================")
    print("simplekNNClassifier.train():")
    classifier.train(trainingData, train_true_label)
    print("simplekNNClassifier.test():")
    classifier.test(testData, 5)
    # for check the predicted_labels
    print(classifier.test(testData, 5))
    print("\n\n")


def kdTreeKNNClassifierTests():
    print("kdTreeKNNClassifier Instantiation.")

    print("==============================================================")
    print("Prepare for the dataset Penguin.")
    # readin file
    data = pd.read_csv(penguins)
    # choose columns needed
    data = data.iloc[:, [2, 9, 10, 11, 12]]
    # drop nas
    data = data.dropna()
    # check the datasets
    data["Species"].value_counts()
    # check the datasets
    data.describe()
    # split into train set and test set
    training = data.sample(frac=0.8, random_state=200)
    test = data.drop(training.index)
    # set for training Data without labels
    trainingData = training.iloc[:, 1:6]
    # training data labels
    train_true_label = training.iloc[:, 0]
    # set for testing Data without labels
    testData = test.iloc[:, 1:6]
    # testing Data labels, will not be used in KNN, but will be used in Experiment
    test_true_label = test.iloc[:, 0]
    print("==============================================================")

    classifier = kdTreeKNNClassifier()
    print("===========================================================")
    print("Check member methods.")
    print("Check that train and test methods have been override.")
    print("===========================================================")
    print("kdTreekNNClassifier.train():")
    classifier.train(trainingData, train_true_label)
    print("kdTreekNNClassifier.test():")
    classifier.test(testData)
    # for check the predicted_labels
    print(classifier.test(testData))
    print("\n\n")


# =====================================================================
# Testing Classifier Class
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# =====================================================================
from Experiment import Experiment


def ExperimentTests():
    print("Experiment class instantiation.")

    penguins = "QuantData/penguins_lter.csv"
    data = pd.read_csv(penguins)
    data = data.iloc[:, [2, 9, 10, 11, 12]]
    data = data.dropna()
    labels = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    lst_classifier = ["simpleKNNClassifier", "simpleKNNClassifier"]

    experiment = Experiment(data, labels, lst_classifier)
    print("==============================================================")
    print("Check member methods.")
    print("==============================================================")
    print("Now call Experiment.runCrossVal(k):")
    # experiment.runCrossVal(5)
    print(experiment.runCrossVal(5))
    print("==============================================================")
    print("Now call Experiment.score:")
    experiment.score()
    print("==============================================================")
    print("Now call Experiment.confusionMatrix.")
    experiment.confusionMatrix(5)
    print("==============================================================")
    print("Now call Experiment.ROC.")
    # saving all data points for ROC curves, since we plot three lines,
    # 6 list of values are needed
    tpr_a = []  # step:1   space:n
    fpr_a = []  # step:1   space:n
    tpr_c = []  # step:1   space:n
    fpr_c = []  # step:1   space:n
    tpr_g = []  # step:1   space:n
    fpr_g = []  # step:1   space:n
    # calculate each of these points
    for k in range(1, 10):
        experiment.confusionMatrix(
            k
        )  # step:O(n^3) * n   space:O(n^2) * n from Experiment.confusion_matrix
        tpr_a.append(experiment.tpr_a)  # step: 21n space: 3n
        fpr_a.append(experiment.fpr_a)  # step: 33n space: 3n
        tpr_c.append(experiment.tpr_c)  # step: 33n space: 3n
        fpr_c.append(experiment.fpr_c)  # step: 33n space: 3n
        tpr_g.append(experiment.tpr_g)  # step: 33n space: 3n
        fpr_g.append(experiment.fpr_g)  # step: 33n space: 3n
    # saving all points into a list
    tpr = [tpr_a, tpr_c, tpr_g]  # step: 3 space: 0
    fpr = [fpr_a, fpr_c, fpr_g]  # step: 3 space: 0
    # get the start points and end points
    for i in tpr:  # step: n space: n
        i.append(0)  # step: n space: n
        i.append(1)  # step: n space: n
        i.sort()  # step: nlogn space: 0
    for j in fpr:  # step: n space: n
        j.append(0)  # step: n space: n
        j.append(1)  # step: n space: n
        j.sort()  # step: nlogn space: 0
    # plot the curve
    experiment.ROC(tpr, fpr)
    print("\n\n\n")


def main():
    # DataSetTests()
    # TimeSeriesDataSetTests()
    # TextDataSetTests()
    # QuantDataSetTests()
    # QualDataSetTests()
    # TransactionDataSetTests()
    HeterogenousDataSetTests()

    # ClassifierAlgorithmTests()
    # simplekNNClassifierTests()
    kdTreeKNNClassifierTests()
    # ExperimentTests()


if __name__ == "__main__":
    main()
