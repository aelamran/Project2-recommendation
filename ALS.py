# coding: utf-8


from helpers_pj import *
from helpers_pj import load_data

# Loading of the Training set:

path_dataset = 'data/data_train.csv'
ratings = load_data(path_dataset)

# Loading of the Submission Set:

test_rating = load_data('data/sample_submission.csv')
test_rating.shape

"""Notice that the index of items and users have been inverted
for sake of ease, basic matrix operations from linear algebra allows
to obtain the correct values and a coherent Matrix.
"""


# Splitting of the data in accordance to the previous comment:

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    # split the ratings to training data and test data.

    # set seed
    np.random.seed(988)

    # select user and item based on the condition, which is set at 0 to not loose insights
    valid_users = np.where(num_items_per_user >= 0)[0]
    valid_items = np.where(num_users_per_item >= 0)[0]
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


# To Perform the ALS model, one needs to initialize the parameters for the Matrix Factorization

def init_MF(train, num_features):
    """
    Source: Lab 10 Solutions
    """
    num_users, num_items = train.shape
    user_features = np.random.rand(num_features, num_users)
    item_features = np.random.rand(num_features, num_items)

    # Sum up each ratings for each movie
    item_sum = train.sum(axis=0)
    item_nnz = train.getnnz(axis=0)

    # set the first item features to the sum of the ratings divided by the number of nonzero items
    for item_index in range(num_items):
        item_features[0, item_index] = int(item_sum[0, item_index] / item_nnz[item_index])

    return user_features, item_features


# Below is the function that updates the user-feature matrix Z:

def update_user_feature(train, item_feat, num_feat, lambda_user, nz_user_itemindices):
    num_users = train.shape[0]
    user_feat = np.zeros((num_feat, num_users))

    for user, items in nz_user_itemindices:
        W = item_feat[:, items]

        A = W @ W.T + lambda_user * sp.eye(num_feat)
        b = W @ train[user, items].T

        x = np.linalg.solve(A, b)

        user_feat[:, user] = np.copy(x.T)

    return user_feat


# Below is the function that updates the item-feature matrix Z:

def update_item_feature(train, user_feat, num_feat, lambda_item, nz_item_userindices):
    num_items = train.shape[1]
    item_feat = np.zeros((num_feat, num_items))

    for item, users in nz_item_userindices:
        W = user_feat[:, users]

        A = W @ W.T + lambda_item * sp.eye(num_feat)
        b = W @ train[users, item]

        x = np.linalg.solve(A, b)

        item_feat[:, item] = np.copy(x.T)

    return item_feat


# RMSE Computing function:

def compute_error(data, user_feat, item_feat, nz):
    mse = 0
    pred = np.dot(user_feat.T, item_feat)

    for row, col in nz:
        mse += np.square((data[row, col] - pred[row, col]))

    return np.sqrt(mse / len(nz))


# Implementation of the ALS Algorithm:
def calculate_als(train, test, test_ratings, seed=988, num_feat=10, m_iter=18, lambda_user=1., lambda_item=0.001,
                  change=1, stop_criterion=1e-8):
    """
    Use Alternating Least Squares (ALS) algorithm to generate predictions,
    for further details, please refer to the report
    """
    error_list = [0]
    itr = 0
    np.random.seed(seed)

    # Initialization of W and Z:
    user_feat, item_feat = init_MF(train, num_feat)

    # Groupping of indices by row or column index
    nz_train, nz_user_itemindices, nz_item_userindices = build_index_groups(train)

    while change > stop_criterion and itr < m_iter:
        # Update W and Z
        user_feat = update_user_feature(train, item_feat, num_feat, lambda_user, nz_user_itemindices)
        item_feat = update_item_feature(train, user_feat, num_feat, lambda_item, nz_item_userindices)

        rmse = compute_error(train, user_feat, item_feat, nz_train)
        print('iteration # %d' % (itr + 1))
        print("RMSE on training set: {}.".format(rmse))
        change = np.fabs(rmse - error_list[-1])
        error_list.append(rmse)
        itr = itr + 1

    # Calculation of the RMSE score:
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_feat, item_feat, nnz_test)
    print("Test RMSE after running ALS: {s}".format(s=rmse))

    num_users = test_ratings.shape[0]
    num_items = test_ratings.shape[1]
    pred_als = sp.lil_matrix((num_users, num_items))

    # Multiplication of the 2 matrix to recover the original
    for user in range(num_users):
        for item in range(num_items):
            item_info = item_feat[:, item]
            user_info = user_feat[:, user]
            pred_als[user, item] = user_info.T.dot(item_info)

    return pred_als
