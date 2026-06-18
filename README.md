# Dataset Toolbox

A data-processing and machine-learning toolkit I wrote from scratch for
Georgetown ANLY 555. The algorithms (median filtering, KNN, KD-Tree search,
Apriori, cross-validation, confusion matrix, ROC) are written by hand instead of
calling scikit-learn. Each method has a note on its time/space complexity.

## Project structure

| File                     | Layer       | What it does                                      |
| ------------------------ | ----------- | ------------------------------------------------- |
| `DataSet.py`             | Data        | Read, clean and explore six dataset types         |
| `ClassifierAlgorithm.py` | Algorithm   | Two KNN classifiers (brute-force and KD-Tree)     |
| `Experiment.py`          | Experiment  | Cross-validation, accuracy, confusion matrix, ROC |
| `test.py`                | Entry point | Demo calls for every class                        |

### Data layer

`DataSet` is the base class with a shared interface (`readFromCSV`, `clean`,
`explore`). Six subclasses override `clean` and `explore`:

- **TimeSeriesDataSet** — median-filter denoising, then line and box plots.
- **TextDataSet** — tokenize and remove stopwords, then a word cloud and the
  top-10 word frequencies.
- **QuantDataSet** — fill missing values with the column mean, then line and box
  plots.
- **QualDataSet** — fill missing values with the mode or median, then a histogram.
- **TransactionDataSet** — build a 0/1 transaction matrix and run Apriori to find
  frequent itemsets, then report support/confidence/lift as `Rule` objects.
- **HeterogenousDataSet** — keep a qualitative and a quantitative dataset together
  and call each one in turn.

### Algorithm layer

`ClassifierAlgorithm` is the base class (`train` / `test`):

- **simplekNNClassifier** — brute force: Euclidean distance to every training
  point, majority vote of the nearest _k_.
- **kdTreeKNNClassifier** — builds a KD-Tree (`Tree` helper class) and searches it
  recursively, skipping branches that cannot hold a closer point.

### Experiment layer

`Experiment` holds the dataset, labels and a list of classifiers:

- `runCrossVal(k)` — k-fold cross-validation
- `score()` — accuracy table
- `confusionMatrix(k)` — confusion matrix plus per-class TPR/FPR
- `ROC(tpr, fpr)` — ROC curves

The classification experiments use the penguins dataset (3 species).

## Requirements

```
pandas
numpy
matplotlib
nltk
wordcloud
```

NLTK also needs the stopwords data:

```python
import nltk
nltk.download("stopwords")
```

## Data layout

The scripts expect these CSV files (not included in the repo):

```
TimeSeriesData/ptbdb_normal.csv
TextData/yelp.csv
QuantData/Sales_Transactions_Dataset_Weekly.csv
QuantData/penguins_lter.csv
QualData/multiple_choice_responses.csv
TransactionData/groceries.csv
```

## Usage

```bash
python test.py
```

The test functions inside `main()` are commented out individually; uncomment the
ones you want to run. The `explore()` methods open matplotlib windows.

## Known limitations

- `Experiment.confusionMatrix` assumes exactly 3 classes (set up for the penguins
  dataset).
- `build_transaction_matrix` runs in O(n^2).
- Testing is limited to the demo functions in `test.py`.
