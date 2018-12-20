from itertools import groupby
import numpy as np
import scipy.sparse as sp
import csv


def pandas_lod_data(path):
    """

    :param path: the path of the data we want to load
    :return: pandas dataframe of our data loaded
    """
    return pd.read_csv(path)


def read_txt(path):
    """

    :param path: the path of the text we want to read
    :return: the desired path
    """
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """

    :param path_dataset: the path of our dataset
    :return: the dataset preprocessed
    """
    """Load data in text format, one rating per line, as in the crowdAI competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """

    :param data: The data we wish to preprocess
    :return: data preprocessed as desired
    """
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        """

        :param line: One line of the dataset
        :return: the row number, column number and finally the rating
        """
        pos, rating = line.split(',')
        row, col = pos.split("_")
        # row contains user and column contains item
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        """

        :param data: Our data
        :return: some basic statistics such as max and min over the rows and columns
        """
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of users: {}, number of items: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating

    return ratings


def group_by(data, index):
    """

    :param data: The data we want to group
    :param index: Our data grouped by the given index
    :return:
    """
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """

    :param train: Our train data
    :return: Non zero data
    """
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """

    :param real_label:  The actual rating
    :param prediction: The predicted rating
    :return:Mean squared error between prediction and real_label
    """
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def splitting(df, column='Id'):
    """

    :param df: The dataframe we want to split using pandas (a different split than only numpy and scipy)
    :param column: the column we split ont
    :return: dataframe containing user_ids, movie_ids and ratings
    """
    '''
        df : the dataframe to split
        column : the column containing the data to split, by default it is the Id column
    '''
    output = df[column].str.split('(\d+)([A-z]+)(\d+)', expand=True)
    output = output.loc[:, [1, 3]]
    output.rename(columns={1: 'userid', 2: 'y', 3: 'movieid'}, inplace=True)
    output['userid'] = output['userid'].astype(float)
    output['movieid'] = output['movieid'].astype(float)
    return output


def split_data(ratings, split=0.1):
    """

    :param ratings: The data we want to split
    :param split: the proportion of the test set
    :return: train and test sets.
    """
    """
    Source: Lab 10 Solutions
    split the ratings to training data and test data.
    """

    # set seed
    np.random.seed(988)

    nz_users, nz_items = ratings.nonzero()

    # create sparse matrices to store the data
    num_rows, num_cols = ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    for item in set(nz_items):
        row, col = ratings[:, item].nonzero()
        selects = np.random.choice(row, size=int(len(row) * split))
        non_selects = list(set(row) - set(selects))

        train[non_selects, item] = ratings[non_selects, item]
        test[selects, item] = ratings[selects, item]

    return train, test


def create_csv_submission(test_data_path, output_path, predictions):
    """

    :param test_data_path: the path of the test data
    :param output_path: the path where we want to output our data
    :param predictions: the predictions we want to submit
    :return:
    """
    """create csv submission for the test data using the predictions."""

    def deal_line(line):
        """

        :param line: The line we want to deal with
        :return: row, columns index
        """
        row_col_id, _ = line.split(',')
        row, col = row_col_id.split("_")
        # row contains user and column contains item
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row) - 1, int(col) - 1, row_col_id

    with open(test_data_path, "r") as f_in:
        test_data = f_in.read().splitlines()
        fieldnames = test_data[0].split(",")
        test_data = test_data[1:]

    with open(output_path, 'w') as f_out:
        writer = csv.DictWriter(f_out, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for line in test_data:
            user, item, user_item_id = deal_line(line)
            prediction = predictions[user, item]
            writer.writerow({
                fieldnames[0]: user_item_id,
                fieldnames[1]: prediction
            })
