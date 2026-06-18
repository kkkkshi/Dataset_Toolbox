# Summary of Object-Oriented Toolbox Design

**Author:** Ke Shi — Georgetown ANLY 555

## Overview

The Dataset Toolbox is a data-processing and machine-learning toolkit I wrote
from scratch over the semester. Instead of calling a library like scikit-learn,
I implemented the algorithms myself: median filtering, k-nearest-neighbour
classification, a KD-Tree for nearest-neighbour search, Apriori for association
rules, plus cross-validation, the confusion matrix and the ROC curve. Each
method also carries a note on its time and space complexity.

## Class Hierarchy

There are two main inheritance chains, an experiment driver, and a couple of
small helper classes.

The data side starts from a base class `DataSet`, which sets up the shared
methods `readFromCSV`, `clean` and `explore`. Six subclasses override `clean`
and `explore` for their own kind of data:

- `TimeSeriesDataSet` cleans the data with a median filter.
- `TextDataSet` tokenizes the text and drops stopwords.
- `QuantDataSet` fills missing values with the column mean.
- `QualDataSet` fills missing values with the mode or the median.
- `TransactionDataSet` builds a 0/1 transaction matrix and runs Apriori to find
  association rules, storing each one as a `Rule` object.
- `HeterogenousDataSet` holds a qualitative and a quantitative dataset together
  and calls each of them in turn.

The classifier side starts from a base class `ClassifierAlgorithm` with `train`
and `test`. Two subclasses do the actual k-nearest-neighbour work:

- `simplekNNClassifier` measures the Euclidean distance to every training point
  and takes the majority vote of the nearest k.
- `kdTreeKNNClassifier` builds a KD-Tree (with the `Tree` helper class) and walks
  it recursively, skipping branches that cannot hold a closer point.

`Experiment` ties it together. It keeps a dataset, its labels and a list of
classifiers, and runs `runCrossVal` for k-fold cross-validation, `score` for
accuracy, `confusionMatrix` for the confusion matrix and per-class TPR/FPR, and
`ROC` for the curves. The classification tests run on the penguins dataset, which
has three species.

## Implementation Decisions

Most of the design comes from object-oriented programming. Inheritance lets the
datasets and classifiers share one interface, so the test code can call `clean`,
`explore`, `train` and `test` without caring about the exact type, and each
algorithm keeps its own state inside its object.

When there was a faster way to do something, I used it. The clearest example is
the KD-Tree classifier: it builds the tree once and prunes branches while
searching, instead of checking the distance to every point like the brute-force
version.

## Reflection

### What went well

- **Readability.** The code follows PEP-8, every class and method has a
  docstring, and the comments explain what each step does.
- **Consistency.** One base class with the same method names everywhere makes the
  project easy to follow.
- **Speed.** I kept looking for ways to cut the running time, mostly in the
  KD-Tree classifier.

### What could be better

- Some methods hard-code column indices or the number of classes to fit the data
  I was using. For example, `confusionMatrix` assumes exactly three categories.
- A few methods are still slow. `build_transaction_matrix` in
  `TransactionDataSet` runs in O(n^2).
- I only wrote one demo function per class. Given more time I would add real test
  cases so the code is harder to break.

## Conclusion

The toolbox does what I set out to build, with a clean object-oriented structure,
some real strengths, and a few limitations I am aware of. I plan to keep working
on it.
