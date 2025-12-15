# * DataSet.py
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
import numpy as np
import matplotlib.pyplot as plt
from nltk import RegexpTokenizer, FreqDist
from nltk.corpus import stopwords
from wordcloud import wordcloud, WordCloud
import copy

time_series_data = "TimeSeriesData/ptbdb_normal.csv"
text_data = "TextData/yelp.csv"
quant_data = "QuantData/Sales_Transactions_Dataset_Weekly.csv"
qual_data = "QualData/multiple_choice_responses.csv"
penguins = "QuantData/penguins_lter.csv"
transaction_data = "TransactionData/groceries.csv"


class DataSet:
    """It is a Base Class to construct.

    Four subclass will be inherited, which is
    TimeSeriesDataSet: for TimeSeries data evaluation
    TextDataSet: for Text data evaluation
    QuantDataSet: for Quantitative data evaluation
    QualDataSet: for Qualitative data evaluation
    """

    def __init__(self, filename, filetype="csv"):
        """The Base class DataSet constructor
        decide whether is a csv or other type file

        filename: string
        return: None
        """
        print("The Dataset constructor is build.")

        if filetype == "csv":
            self.data = self.readFromCSV(filename)
        else:
            self.data = self.load(filename)

    def readFromCSV(self, filename):
        """The private readFromCSV function, it will read the CSV to be further process

        filename: string
        return: None
        """
        print("In DataSet, the readFromCSV is build.")
        data = pd.read_csv(filename, header=None)
        return data

    def load(self, filename):
        """The private load function, it will load the CSV to be further process

        filename: string
        return: None
        """
        print("In DataSet, the load is build.")
        with open(filename, "r") as file:
            data = file.read()
        return data

    def clean(self):
        """The clean function, it will clean the dataset to be further process
        Subclass_clean will overwrite this function

        return: None
        """
        print("In DataSet, the clean function is build.")
        print("Will be override in subclasses.")

    def explore(self):
        """The explore function, it will explore the dataset to be shown for the audience
        Subclass_explore will overwrite this function

        return: None
        """

        print("In DataSet, the explore function is build.")
        print("Will be override in subclasses.")


class TimeSeriesDataSet(DataSet):
    """It is a subclass called TimeSeriesDataSet
    and is inherited from the base class DataSet.

    It will be dealing with TimeSeries data set specifically.
    """

    def __init__(self, filename):
        """This is the constructor for TimeSeries Dataset,
        it is inherited from the base class DataSet.
        """
        super().__init__(filename)
        print("In TimeSeries DataSet, the constructor is inherit.")

    def clean(self, window_size=3):
        """This is a clean function for TimeSeries DataSet.
        It will override the clean function from the base class DataSet.
        clean the data by median_filter, which can be chosen from window_size

        return: None
        """
        self.data = self.medium_filter()
        print(self.data)
        print("In TimeSeries Dataset, the clean is rebuild.")

    def explore(self):
        """This is an explore function for TimeSeries DataSet
        It will override the explore function from the base class DataSet
        Making two graphs about time series, one for line chart, one for box plot

        return: None
        """
        fig, ax = plt.subplots()
        self.data.transpose().iloc[:, 0:10].plot(ax=ax)
        ax.set_title("Time Series Line Plot")
        fig, ax = plt.subplots()
        self.data.transpose().iloc[:, 0:10].boxplot(ax=ax)
        ax.set_title("Time Series Box Plot")
        plt.show()
        print("In TimeSeries Dataset, the explore is rebuild.")

    def self_median(self, nums):
        """A self median code to calculate the median of a list,
        giving a list and get the medium of it

        nums: List[float]
        return: float
        """
        for i in range(len(nums) // 2 + 1):
            min_index = i
            for j in range(i + 1, len(nums)):
                if nums[min_index] > nums[j]:
                    min_index = j
            nums[i], nums[min_index] = nums[min_index], nums[i]
        if len(nums) % 2 != 0:
            return nums[len(nums) // 2]
        return (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2

    def medium_filter(self, filter_size=3):
        """A self medium_filter, which take the filter_size to run,
        filter_size is automatically 3, it can be changed to any odd number
        Get the left part value and right part value and do the medium to find the value

        filter_size: int
        return: dataframe
        """
        col_number = self.data.shape[1]
        row_number = self.data.shape[0]
        new_data = self.data
        if filter_size == 1:
            return self.data
        elif filter_size % 2 != 1:
            raise ValueError("Not an odd number")
        else:
            start_col = filter_size // 2
            end_col = col_number - start_col
            part_size = filter_size // 2
            for r in range(row_number):
                for l in range(start_col, end_col + 1):
                    new_data.iloc[r, l] = self.self_median(
                        list(self.data.iloc[r, l - part_size : (l + part_size + 1)])
                    )
        return new_data


class TextDataSet(DataSet):
    """It is a subclass called TextDataSet
    and is inherited from the base class DataSet.

    It will be dealing with Text data set specifically.
    """

    def __init__(self, filename):
        """This is the constructor for Text Dataset,
        it is inherited from the base class DataSet.
        """
        super().__init__(filename)
        self.stop_words = stopwords.words("english")
        print("In Text DataSet, the constructor is inherit.")

    def clean(self, extra_stopwords="", language="english"):
        """This is a clean function for Text DataSet.
        It will override the clean function from the base class DataSet.
        Remove the stopwords from the file
        In extra_stopwords, we can add other stopwords
        In language, we can choose another language's stopwords

        return: None
        """
        col_number = self.data.shape[1]
        row_number = self.data.shape[0]
        stop_words = list(set(stopwords.words("english")))
        if extra_stopwords != "":
            for i in extra_stopwords:
                stop_words.append(i)
        tokenizer = RegexpTokenizer(r"\w+")
        for i in range(row_number):
            for j in range(col_number):
                clean_words = tokenizer.tokenize(self.data.iloc[i, j])
                clean_words = [i for i in clean_words if i not in stop_words]
                clean_words = [i for i in clean_words if len(i) > 3 & len(i) < 11]
                self.data.iloc[i, j] = " ".join(clean_words)
        print(self.data)
        print("In Text Dataset, the clean is rebuild.")
        return self.data

    def all_words(self):
        """Combing all the words of the dataframe for doing the explore graphs

        return: str
        """
        str = ""
        col_number = self.data.shape[1]
        row_number = self.data.shape[0]
        for i in range(row_number):
            for j in range(col_number):
                str += " " + self.data.iloc[i, j]
        return str

    def explore(self):
        """This is an explore function for TimeSeries DataSet.
        It will override the explore function from the base class DataSet.
        It will make two graphs about text data, one for wordcloud and
        another is the frequency distribution for top 10 words

        return: None
        """
        all_word = self.all_words()
        wordcloud = WordCloud().generate(all_word)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        fdist_filtered = FreqDist(all_word.split())
        fdist_filtered.plot(
            10, title="Frequency distribution for 10 most common tokens)"
        )

        print("In Text DataSet, the explore is rebuild.")


class QuantDataSet(DataSet):
    """It is a subclass called QuantData
    and is inherited from the base class DataSet.

    It will be dealing with Quantitative data set specifically.
    """

    def __init__(self, filename):
        """This is the constructor for Quantitative Dataset,
        it is inherited from the base class DataSet.
        """
        super().__init__(filename)
        print("In Quant DataSet, the constructor is inherit.")

    def readFromCSV(self, filename):
        """The private readFromCSV function, it will read the CSV to be further process
        readFromCSV with the header for easy of clean and explore

        filename: string
        return: None
        """
        print("In Quant DataSet, the readFromCSV is rebuild.")
        data = pd.read_csv(filename)
        return data

    def clean(self):
        """This is a clean function for Quantitative DataSet.
        It will override the clean function from the base class DataSet.
        Filling the missing value with mean

        return: None
        """
        print("In Quant Dataset, the clean is rebuild.")
        col_number = self.data.shape[1]
        for i in range(1, col_number):
            mean_column = self.self_mean(list(self.data.iloc[:, i]))
            self.data.iloc[:, i].fillna(mean_column, inplace=True)
        print(self.data)
        return self.data

    def self_mean(self, nums):
        """This is a function for mean of a list

        nums: List[float]
        return: mean of the list
        """
        total = 0
        for i in range(len(nums)):
            total += nums[i]
        mean = total / len(nums)
        return mean

    def explore(self):
        """This is an explore function for Quantitative DataSet.
        It will override the explore function from the base class DataSet.
        Explore two graphs about the quantitative data, one for line plot
        another is for boxplot

        return: None
        """
        self.data.iloc[:, 1:5].plot()
        fig, ax = plt.subplots()
        self.data.iloc[0:25, 0:25].boxplot()
        ax.set_title("Quant data Boxplot")
        plt.show()

        print("In Quant DataSet, the explore is rebuild.")


class QualDataSet(DataSet):
    """It is a subclass called QualData
    and is inherited from the base class DataSet.

    It will be dealing with Qualitative data set specifically.
    """

    def __init__(self, filename):
        """This is the constructor for Qualitative Dataset,
        it is inherited from the base class DataSet.
        """
        super().__init__(filename)
        print("In Qual DataSet, the constructor is inherit.")

    def readFromCSV(self, filename):
        """The private readFromCSV function, it will read the CSV to be further process
        readFromCSV include the header for easy of clean and explore

        filename: string
        return: None
        """
        print("In Qual DataSet, the readFromCSV is rebuild.")
        data = pd.read_csv(filename, low_memory=False)
        return data

    def clean(self, method="median"):
        """This is a clean function for Qualitative DataSet.
        It will override the clean function from the base class DataSet.
        Filling the na's with median or mode
        If don't choose, will automatically use median

        return: None
        """
        print("In Qual Dataset, the clean is rebuild.")
        col_number = self.data.shape[1]
        if method == "mode":
            for i in range(col_number):
                tmp = [x for x in (list(self.data.iloc[:, i])) if pd.isnull(x) == False]
                test = self.mode(tmp)
                self.data.iloc[:, i].fillna(test, inplace=True)
            return self.data
        else:
            for i in range(col_number):
                column_median = (
                    self.data.iloc[:, i]
                    .astype(str)
                    .str.extractall("(\d+)")
                    .fillna("")
                    .astype(int)
                    .median()
                )
                self.data.iloc[:, i].fillna(column_median, inplace=True)
            print(self.data)
            return self.data

    def mode(self, nums):
        """Giving a list of nums and return the mode of the list

        nums: List[int], List[str], List[*]...
        return: int/str/*...
        """
        return max(set(nums), key=nums.count)

    def explore(self):
        """This is an explore function for Qualatitative DataSet.
        It will override the explore function from the base class DataSet.
        Explore two graphs about the qualitative data, one for line plot
        another is for boxplot

        return: None
        """
        new_data = self.data[1:]
        fig, ax = plt.subplots()
        plt.plot(list(new_data.iloc[0:50, 0]))
        ax.set_title("Qualitative Line Plot")
        plt.show()
        plt.hist(list(new_data.iloc[0:100, 2]), density=True)
        ax.set_title("Qualitative Hist Plot")
        plt.show()


class TransactionDataSet(DataSet):
    """It is a subclass called TransactionData
    and is inherited from the base class DataSet.

    It will be dealing with Transaction data set specifically.
    """

    def __init__(self, filename):
        """This is the constructor for Transaction Dataset,
        it is inherited from the base class DataSet.

        original_item_set: List
        final_item_set: List
        """
        super().__init__(filename)
        # saving the first item possible items
        self.original_item_set = []
        # saving all possible items
        self.final_item_set = list()
        print("In Transaction DataSet, the constructor is inherit.")

    def readFromCSV(self, filename):
        """The private readFromCSV function, it will read the CSV to be further process

        filename: string
        return: None
        """
        print("In Transaction DataSet, the readFromCSV is rebuild.")
        # nrows can be changed, the data is too large to run. If use all data,
        # the support will be very low
        self.data = pd.read_csv(filename, nrows=100)

    def transaction_matrix(self):
        """Transfer the original csv into a countvectorizer dataframe by own written code.
        Each column is a product, if 1 appears, the customer get the item, 0 for otherwise

        return: None
        """
        # count how many different items appear in the dataframe
        item_set = set()
        for index, row in self.data.iterrows():
            item_set.update(row)
        item_set = {x for x in item_set if pd.notna(x)}
        new_columns = len(item_set)
        new_rows = self.data.shape[0]

        # build a countvectorizer dataframe by own code, 1 for appear, 0 for not
        self.transaction_matrix = pd.DataFrame(
            np.zeros([new_rows, new_columns]), columns=item_set
        )
        for index, row in self.data.iterrows():
            for item in row:
                if item == np.nan:
                    continue
                self.transaction_matrix.loc[index][item] = 1

    def item_lists(self, prev_item_set):
        """The apriori method is implemented, prev_item_set is the previous item set, if previous set has
        one item per list, it will return two item per list for next level

        prev_item_set: List[List]
        return: List[List]
        T(n) = 3 + 3n + 11n^2 + 2n^2*logn + 7n^3  = O(n^3)
        S(n) = 7n + 2n^2  = O(n^2)
        """
        # saving new produced items
        new_item_set = list()  # step:1   space:n
        # two situation appears, one if the prev_item_set is List[string](first circulation),
        # the other situation is prev_item_set is List[List] (two or more circulation)
        for i in prev_item_set:  # step:n   space:n
            if isinstance(i, str):  # step:n   space:3n
                # built the new item set
                rows = self.transaction_matrix[
                    self.transaction_matrix[i] == 1
                ]  # step:n* (n^2)   space:n^2
                rows = (
                    rows.sum().sort_values(ascending=False).reset_index()
                )  # step:n^2*logn   space:0
                rows.rename(
                    columns={
                        rows.columns[0]: "item_name",
                        rows.columns[1]: "count",
                    },  # step:n*n   space:0
                    inplace=True,
                )
                rows = rows[rows["count"] >= 5]  # step:n*(n^2)   space:0
                rows = rows[rows["item_name"] != i]  # step:n*(n^2)   space:0
                items = rows["item_name"].tolist()  # step:n*(n^2)   space:n
                for j in items:  # step:n*n   space:0
                    if [j, i] in new_item_set:  # step:n*n   space:0
                        continue
                    new_item_set.append([i, j])  # step:n*n   space:0
            else:
                for k in i:  # step:n   space:n
                    # check each item in the matrix to be 1 and built the new item set
                    rows = self.transaction_matrix[
                        self.transaction_matrix[k] == 1
                    ]  # step:n*(n^2)   space:n^2
                rows = (
                    rows.sum().sort_values(ascending=False).reset_index()
                )  # step:n^2logn   space:0
                rows.rename(
                    columns={
                        rows.columns[0]: "item_name",
                        rows.columns[1]: "count",
                    },  # step:n*n   space:0
                    inplace=True,
                )
                rows = rows[rows["count"] >= 5]  # step:n*(n^2)   space:0
                items = rows["item_name"].tolist()  # step:n*(n^2)   space:n
                for j in items:  # step:n*n   space:0
                    if (j in i) or [j, i] in new_item_set:  # step:2n*n   space:0
                        continue
                    lst = copy.deepcopy(i)  # step:n*n   space:n
                    lst.append(j)  # step:n*n   space:0
                    new_item_set.append(lst)  # step:n*n   space:0
        return new_item_set  # step:2   space:0

    def itemset(self):
        """Using the previous function item_lists, to circular calculate the FP-growth algorithm
        final_item_set will all possible categories that has high support and might fit the supportThreshold > 0.25
        condition

        return: None
        T(n) = 1 + 5n + n^2 + nlogn + [2 * O(n^3) from item_list function]
        S(n) = n + n^2 + [2 * O(n^2) from item_list function]
        """
        # filter out the items that could be supporthreshold > 0.25
        item_count = (
            self.transaction_matrix.sum()
            .sort_values(ascending=False)
            .reset_index()  # step:nlogn   space:n
        )
        item_count.rename(  # step:n   space:0
            columns={
                item_count.columns[0]: "item_name",
                item_count.columns[1]: "count",
            },
            inplace=True,
        )
        item_count = pd.DataFrame(data=item_count)  # step:n^2   space:n^2
        # first time filter, for one item
        self.original_item_set.extend(  # step:1   space:n
            item_count[item_count["count"] > 10][
                "item_name"
            ].tolist()  # step:n   space:0
        )
        # second time filter, for two items together
        # step:O(n^3) from item_list function  space:O(n^2) from item_list function
        new_set = self.item_lists(self.original_item_set)
        # third time filter, for three items together, where it will reach the minimum support I constructed
        # fourth time could be constructed, but it's too low
        # step:O(n^3) from item_list function  space:O(n^2) from item_list function
        new_set2 = self.item_lists(new_set)
        for i in self.original_item_set:  # step:n   space:0
            self.final_item_set.append([i])
        # combine all three times items together
        self.final_item_set.extend(new_set)  # step:n   space:0
        self.final_item_set.extend(new_set2)  # step:n   space:0

    def clean(self):
        """Clean method for Transaction data, overwrite the function in base class
        Call transaction_matrix function to form a matrix
        Call itemset function to form a final itemset

        return: None
        To conclude: the Apriori algorithm:
        T(n) = O(n^3)
        S(n) = O(n^2)
        """
        # form the transaction matrix
        self.transaction_matrix()
        # built the itemset
        self.itemset()

    def support(self, left, right):
        """Support: Transaction contain both X and Y / Total number of transaction
        get the left item appearances and right item appearances to calculate the support

        left: string/List
        right: string/List
        return: float
        """
        # left and right can be a string or a list
        if isinstance(left, str):
            numerator = self.transaction_matrix[self.transaction_matrix[left] == 1]
        else:
            for i in left:
                numerator = self.transaction_matrix[self.transaction_matrix[i] == 1]
        if isinstance(right, str):
            numerator = self.transaction_matrix[self.transaction_matrix[right] == 1]
        else:
            for j in right:
                numerator = self.transaction_matrix[self.transaction_matrix[j] == 1]
        # calculate the support based on counts
        denominator = self.transaction_matrix.shape[0]
        numerator = numerator.shape[0]
        supp = numerator / denominator
        return supp

    def confidence(self, left, right):
        """Confidence: Transaction contain both X and Y / Transaction contain X
        get the left item appearances and right item appearances to calculate the confidence

        left: string/List
        right: string/List
        return: float
        """
        # left and right can be a string or a list
        if isinstance(left, str):
            numerator = self.transaction_matrix[self.transaction_matrix[left] == 1]
        else:
            for i in left:
                numerator = self.transaction_matrix[self.transaction_matrix[i] == 1]
        denominator = numerator.shape[0]
        if isinstance(right, str):
            numerator = self.transaction_matrix[self.transaction_matrix[right] == 1]
        else:
            for j in right:
                numerator = self.transaction_matrix[self.transaction_matrix[j] == 1]
        # calculate the confidence based on formula
        numerator = numerator.shape[0]
        conf = numerator / denominator
        return conf

    def lift(self, left, right):
        """Lift: Transaction contains both X and Y / ((Transaction contains X) * (Transaction contains Y))
        get the left item appearances and right item appearances to calculate the lift

        left: string/List
        right: string/List
        return: float
        """
        # left and right can be string or list
        if isinstance(left, str):
            numerator = self.transaction_matrix[self.transaction_matrix[left] == 1]
        else:
            for i in left:
                numerator = self.transaction_matrix[self.transaction_matrix[i] == 1]
        if isinstance(right, str):
            numerator = self.transaction_matrix[self.transaction_matrix[right] == 1]
        else:
            for j in right:
                numerator = self.transaction_matrix[self.transaction_matrix[j] == 1]
        numerator = numerator.shape[0]

        if isinstance(left, str):
            denominator_x = self.transaction_matrix[self.transaction_matrix[left] == 1]
        else:
            for i in left:
                denominator_x = self.transaction_matrix[self.transaction_matrix[i] == 1]
        denominator_x = denominator_x.shape[0]
        if isinstance(right, str):
            denominator_y = self.transaction_matrix[self.transaction_matrix[right] == 1]
        else:
            for j in right:
                denominator_y = self.transaction_matrix[self.transaction_matrix[j] == 1]
        # calculate the lift based on formula
        denominator_y = denominator_y.shape[0]
        lift = numerator / (denominator_x * denominator_y)
        return lift

    def explore(self, supportThreshold=0.25):
        """given the supportThreshold, to make the transaction items' support, confidence and lift
        in this format: ['A'] ==> ['B'] | Sup =  | Conf =  | Lift =

        return: None
        """
        # saving all the rules
        rules = []
        for i in self.final_item_set:
            for j in self.final_item_set:
                state = True
                if i != j and len(i) == 1 and len(j) == 1:
                    rules.append(
                        Rule(
                            i,
                            j,
                            self.support(i, j),
                            self.confidence(i, j),
                            self.lift(i, j),
                        )
                    )
                if len(j) - len(i) == 1:
                    for k in i:
                        if k not in j:
                            state = False
                            break
                    if state == True:
                        rules.append(
                            Rule(
                                i,
                                j,
                                self.support(i, j),
                                self.confidence(i, j),
                                self.lift(i, j),
                            )
                        )
        # sort the rules based on support value
        for i in range(1, len(rules)):
            for j in range(i - 1):
                if rules[j].support < rules[i].support:
                    rules[j], rules[i] = rules[i], rules[j]
        # check if the support is larger than the supportThreshold, then print out
        for k in range(10):
            if rules[k].support >= supportThreshold:
                print(rules[k])


class Rule:
    """It is a Base class for saving the data of left, right, support, confidence and lift
    It's a clear way to show the data
    """

    def __init__(self, left, right, support, confidence, lift):
        """The Base class Rule constructor

        left: string
        right: string
        support: float
        confidence: float
        lift: float
        return: None
        """
        self.left = left
        self.right = right
        self.support = support
        self.confidence = confidence
        self.lift = lift

    def __str__(self):
        """The output format of Rule

        return: string
        """
        return f"{self.left} ==> {self.right} | Sup = {self.support} | Conf = {self.confidence} | Lift = {self.lift}"


class HeterogenousDataSet(DataSet):
    """It is a subclass called HeterogenousDataset
    and is inherited from the base class DataSet.

    It will be dealing with two or more different data set specifically.
    """

    def __init__(self, filenames):
        """This is the constructor for Heterogenous Dataset,
        it is inherited from the base class DataSet.

        filenames: List[string]
        return: None
        """
        self.filenames = filenames
        self.dataset = []

    def readFromCSV(self):
        """The private readFromCSV function, it will read the CSV to be further process
        the first one will be Qualitative dataset and the second will be Quantitative dataset

        return: None
        """
        self.dataset.append(QualDataSet(self.filenames[0]))
        self.dataset.append(QuantDataSet(self.filenames[1]))

    def clean(self):
        """Call each dataset's clean function

        return: None
        """
        self.dataset[0].clean()
        self.dataset[1].clean()

    def explore(self):
        """Call each dataset's explore function
        """
        self.dataset[0].explore()
        self.dataset[1].explore()

    def select(self, index):
        """Call by the index of the datasets, start with 0

        index: Int
        return: DataFrame
        """
        if index >= 0 and index <= len(self.dataset):
            self.selected = self.dataset[index]
        else:
            ValueError("Index out of range")
        return self.selected
