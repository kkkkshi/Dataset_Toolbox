# Final Project Report: An Object-Oriented Data Science Toolbox

**Author:** Ke Shi
**Course:** ANLY 555, Georgetown University

## 1. Overview

This project is a data-processing and machine-learning toolbox that I built up
over the semester, one deliverable at a time. The goal was to implement the
common steps of a data-science workflow from scratch, without leaning on a
library like scikit-learn, so that the mechanics of each algorithm are visible
in the code. The toolbox can read six different kinds of datasets, clean and
visualize each of them, classify data with two different k-nearest-neighbour
methods, and evaluate those classifiers with cross-validation, a confusion
matrix, and ROC curves. Every method carries a comment with its estimated time
and space complexity.

The code is organized into four files:

| File | Role |
|------|------|
| `DataSet.py` | Reading, cleaning, and exploring six dataset types |
| `ClassifierAlgorithm.py` | Two KNN classifiers and the KD-Tree helper |
| `Experiment.py` | Cross-validation, accuracy, confusion matrix, ROC |
| `test.py` | A driver that exercises every class |

## 2. Design

The design is built around two class hierarchies and follows the standard
object-oriented ideas of inheritance, polymorphism, and encapsulation.

**The data hierarchy** starts from a base class `DataSet`. It defines the shared
interface — `readFromCSV`, `load`, `clean`, and `explore` — and the constructor
decides how to read the file. Six subclasses inherit from it and override `clean`
and `explore` to fit their own kind of data. Because they all share the same
method names, the driver code can treat any dataset the same way, which is the
practical payoff of polymorphism.

**The classifier hierarchy** starts from a base class `ClassifierAlgorithm` with
`train` and `test`. Two subclasses implement KNN in different ways. There are
also two small helper classes: `Tree`, which holds a single node of the KD-Tree,
and `Rule`, which stores one association rule (left side, right side, support,
confidence, lift) so that the output is easy to read.

## 3. Modules

### 3.1 Data layer (`DataSet.py`)

- **TimeSeriesDataSet** — cleans the signal with a median filter that I wrote by
  hand (`self_median` plus `medium_filter`), then plots a line chart and a box
  plot.
- **TextDataSet** — tokenizes the text, removes stopwords, and keeps words of a
  sensible length; it then shows a word cloud and the ten most frequent tokens.
- **QuantDataSet** — fills missing values with the column mean (`self_mean`,
  which ignores NaNs) and plots line and box plots.
- **QualDataSet** — fills missing values with the mode or the median and plots a
  histogram.
- **TransactionDataSet** — builds a 0/1 transaction matrix and mines association
  rules with the Apriori method, reporting support, confidence, and lift.
- **HeterogenousDataSet** — holds a qualitative and a quantitative dataset
  together and dispatches `clean`/`explore` to each, with a `select` method to
  pull one back out.

### 3.2 Classifier layer (`ClassifierAlgorithm.py`)

- **simplekNNClassifier** — the brute-force version. For each test row it
  computes the Euclidean distance to every training row, takes the k closest, and
  returns the majority label.
- **kdTreeKNNClassifier** — builds a KD-Tree once from the training data, then
  answers each query by walking the tree and collecting the k nearest neighbours,
  pruning any branch that cannot contain a closer point. This avoids measuring
  the distance to every point, which is the reason to use a tree in the first
  place.

### 3.3 Experiment layer (`Experiment.py`)

`Experiment` holds a dataset, its labels, and a list of classifiers. It provides:

- `runCrossVal(k)` — k-fold cross-validation.
- `score()` — splits the data, trains, predicts, and reports accuracy.
- `confusionMatrix(k)` — builds the confusion matrix and the per-class true- and
  false-positive rates.
- `ROC(tpr, fpr)` — plots one ROC curve per class.

The classification experiments run on the Palmer penguins dataset, which has
three species and four numeric features.

## 4. Algorithms and complexity

The methods are annotated with rough `T(n)` (time) and `S(n)` (space) estimates.
The main ones are:

| Algorithm | Time | Space |
|-----------|------|-------|
| Median filter (TimeSeries clean) | O(n²) | O(n) |
| Apriori rule mining (Transaction) | O(n³) | O(n²) |
| simple KNN test | O(n²) | O(n) |
| KD-Tree build | O(n²) | O(n log n) |
| KD-Tree test | O(n²) | O(n²) |
| Experiment score / confusionMatrix | O(n³) | O(n²) |
| ROC (drives confusionMatrix over k) | O(n⁴) | O(n³) |

The KD-Tree and the brute-force KNN have the same worst-case bound, but the tree
prunes large parts of the search on well-separated data, so in practice it
usually touches far fewer points.

## 5. Testing

`test.py` has one test function per class that instantiates it and calls its
methods, which is how I checked each deliverable as I went. Beyond that, I
verified the trickier pieces directly:

- The KD-Tree classifier was checked against the brute-force classifier on data
  designed so that the single nearest neighbour and the k-nearest majority
  disagree; both classifiers returned the k-nearest answer, confirming the tree
  search is a true k-NN rather than a 1-NN.
- The cleaning and association-rule code was checked on small hand-built inputs,
  including edge cases such as an all-missing column and items that never appear
  together, to make sure they return sensible values instead of crashing.

## 6. Limitations and future work

- Some methods hard-code assumptions to fit the data I used. The clearest case is
  `confusionMatrix`, which assumes exactly three classes because the penguins
  dataset has three species. A general version would build an N×N matrix.
- A few methods are still expensive. `build_transaction_matrix` is O(n²), and the
  Apriori search grows quickly, which is why the transaction reader only loads a
  slice of the data.
- Testing is mostly demonstration code plus the targeted checks above. A real
  test suite with assertions for each method would make the toolbox easier to
  change safely.

## 7. Conclusion

The toolbox covers the data-science pipeline end to end — reading, cleaning,
visualizing, classifying, and evaluating — with the algorithms written out by
hand and a consistent object-oriented structure tying it together. It has clear
strengths in readability and consistency, and a few known limitations that I
would address next by generalizing the hard-coded parts and lowering the cost of
the heaviest methods.
