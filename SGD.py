# # Machine Learning - Project 2 - 2018
# ## Fatine Benhsain - Tabish Qureshi - Ayyoub El Amrani
# ### Recommender System, SGD

import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp

import warnings

warnings.filterwarnings('ignore')


# The data are stacked into 2 columns with the Id rX_cY. X corresponds to a user and Y corresponds to the movie.
# In order to make a proper analysis, one needs to group users (same X) and the rating (Prediction) on movies (Y).
# For this sake, it is necessary to :
# 1. Unstack the Id and separate X and Y
# 2. Group the same X (users) as rows with corresponding movies (Y) as columns and the rating as argument of the cell.

# We define a method that will split the Id into two columns : _User_ and _Movie_.


def splitting(df, column='Id'):
    """
        df : the dataframe to split
        column : the column containing the data to split, by default it is the Id column
    """
    output = df[column].str.split('(\d+)([A-z]+)(\d+)', expand=True)
    output = output.loc[:, [1, 3]]
    output.rename(columns={1: 'User', 2: 'y', 3: 'Movie'}, inplace=True)
    output['User'] = output['User'].astype(int)
    output['Movie'] = output['Movie'].astype(int)
    output['eval'] = df['Prediction']
    output.pivot(index='Movie', columns='User', values='eval')
    return output


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        ratings: the sparse matrix we want to split to train and test
        p_test: the proportion of data that goes in the test set
        min_num_ratings: 
            used in the case we want all users and items we keep to have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]
    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format(
        (num_rows, num_cols)))

    nz_items, nz_users = valid_ratings.nonzero()

    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test


def init_MF(train, num_features):
    """init the parameter for matrix factorization.
        num_features: the number of features we want to set
    """

    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (RMSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))


def matrix_factorization_SGD(train, test, gamma=0.01, num_features=2, num_epochs=20, lambda_user=0.1, lambda_item=0.7):
    """matrix factorization by SGD.

    num_epochs : number of full passes through the train set
    num_features : number of features we set for our matrices
    lambda_user & lambda_item : regularization parameters
    """
    print('gamma=' + str(gamma) + ' K=' + str(num_features) + ' epochs=' + str(num_epochs) + ' lambda_user= ' + str(
        lambda_user) + ' lambda_item = ' + str(lambda_item))
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(0, num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)

            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)
            if np.math.isnan(item_features[0, d]):
                print(d)

        # rmse = compute_error(train, user_features, item_features, nz_train)
        # print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        # evaluate the test error
        rmse = compute_error(test, user_features, item_features, nz_test)
        print("RMSE on test data: {}.".format(rmse))

        errors.append(rmse)

    return errors, item_features, user_features


def hyper_parameters(train, test, gammas, Ks, num_epochs=20, lambdas_user={0.1}, lambdas_item={0.7}):
    '''
    This method does the grid search to get the best hyper parameters
    :return: the best combination of parameters (with the lowest error)
    '''
    errors = {}
    for g in gammas:
        for k in Ks:
            for lambda_user in lambdas_user:
                for lambda_item in lambdas_item:
                    err, wt, zt = matrix_factorization_SGD(train, test, gamma=g, num_features=k, num_epochs=num_epochs,
                                                           lambda_item=lambda_item, lambda_user=lambda_user)
                    error_min = min(err[1:])
                    errors[(g, k, err.index(error_min), lambda_user, lambda_item)] = error_min
    (gamma_final, k_final, num_epochs_final, lambda_user, lambda_item) = min(errors, key=errors.get)
    return gamma_final, k_final, num_epochs_final, lambda_user, lambda_item


def give_predictions(Wt, Zt):
    '''
    :return: The prediction on the values of the ratings
    '''
    return Wt.T.dot(Zt)


def packing(df, column='Id', user='User', item='Movie', prediction='Prediction'):
    '''
        Method used to put the data in the form rUserId_cItemId
        df : the dataframe to split
        column : the column containing the data to pack, by default it is the Id column
    '''
    output = pd.DataFrame()
    output[column] = 'r' + df[user].astype(str) + '_c' + df[item].astype(str)
    output[prediction] = round(df[prediction]).astype(int)
    output[prediction] = output[prediction].replace(6, 5)
    output[prediction] = output[prediction].replace(0, 1)
    output = output.set_index(column)
    return output


def train_sgd_grid_search(gammas, Ks, num_epochs, lambdas_user, lambdas_item, path='data/data_train.csv',
                          sample_path='data/sample_submission.csv', out_path="submission.csv"):
    '''This method does the grid search on the parameters given and then generates a submission for the best combination,
    using all the methods described before
    '''
    import time
    start_time = time.time()
    raw_data = pd.read_csv(path)
    rating_table = splitting(raw_data)
    rating_table = rating_table.pivot(index='Movie', columns='User', values='eval')

    # We generate a sparse matrix from the dataframe
    ratings_sparse = scipy.sparse.csr_matrix(rating_table.fillna(0).values)
    num_items_per_user = np.array((ratings_sparse != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings_sparse != 0).sum(axis=1).T).flatten()

    valid_ratings, train, test = split_data(
        ratings_sparse, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)
    print("--- %s seconds ---" % (time.time() - start_time))

    # We do the grid search
    gamma_final, k_final, num_epochs_final, lambda_user, lambda_item = hyper_parameters(train, test, gammas, Ks,
                                                                                        num_epochs, lambdas_user,
                                                                                        lambdas_item)
    print(gamma_final)
    print(k_final)
    print(num_epochs_final)
    print(lambda_item)
    print(lambda_user)
    errors, Wt, Zt = matrix_factorization_SGD(train, test, gamma=gamma_final, num_features=k_final,
                                              num_epochs=num_epochs_final,
                                              lambda_user=lambda_user, lambda_item=lambda_item)

    submission = pd.read_csv(sample_path)
    test_submission = splitting(submission)
    prediction = give_predictions(Wt, Zt)
    test_submission['Prediction'] = prediction[test_submission.Movie - 1, test_submission.User - 1]
    result = packing(test_submission)
    result.to_csv(out_path)
    print("--- %s seconds ---" % (time.time() - start_time))


#train_sgd_grid_search(gammas={0.009}, Ks={15}, num_epochs=25, lambdas_user={0.025}, lambdas_item={0.08})


def train_sgd(gamma_final, k_final, num_epochs_final, lambda_user, lambda_item, path='data/data_train.csv',
              sample_path='data/sample_submission.csv', out_path="submission.csv"):
    '''
    This method implements all the above methods to compute the SGD algorithm,and then generates a submission for the
    best combination, given one precise set of parameters.
    '''
    import time
    start_time = time.time()
    raw_data = pd.read_csv(path)
    rating_table = splitting(raw_data)
    rating_table = rating_table.pivot(index='Movie', columns='User', values='eval')
    ratings_sparse = scipy.sparse.csr_matrix(rating_table.fillna(0).values)
    num_items_per_user = np.array((ratings_sparse != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings_sparse != 0).sum(axis=1).T).flatten()

    valid_ratings, train, test = split_data(
        ratings_sparse, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(gamma_final)
    print(k_final)
    print(num_epochs_final)
    print(lambda_item)
    print(lambda_user)
    errors, Wt, Zt = matrix_factorization_SGD(train, test, gamma=gamma_final, num_features=k_final,
                                              num_epochs=num_epochs_final,
                                              lambda_user=lambda_user, lambda_item=lambda_item)

    submission = pd.read_csv(sample_path)
    test_submission = splitting(submission)
    prediction = give_predictions(Wt, Zt)
    test_submission['Prediction'] = prediction[test_submission.Movie - 1, test_submission.User - 1]
    result = packing(test_submission)
    result.to_csv(out_path)
    print("--- %s seconds ---" % (time.time() - start_time))


train_sgd(gamma_final=0.009, k_final=5, num_epochs_final=50, lambda_user=0.015, lambda_item=0.05)
