# Important imports
import pandas as pd
import torch
from helpers_pj import *
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
torch.set_default_tensor_type('torch.DoubleTensor')


# Loading train data
print("---------LOADING DATA-----------")
raw_data = pd.read_csv('data/data_train.csv')


def create_input_for_spotlight(user_id, movie_id, ratings):
    """

    :param user_id: a list containing the id of users
    :param movie_id: a list containing the id of movies
    :param ratings: a list containing the corresponding ratings
    :return: Interaction Object (a useful object containing users, movies and ratings)
    """
    return Interactions(user_id, movie_id, ratings)

def get_train_test_set(interaction):
    """

    :param interaction: Our interaction object (input data)
    :return: Our interaction object (input data) splitted into train and test sets.
    """
    return random_train_test_split(interaction)


def create_model(loss, k, number_epochs, batch_size, l2_penal, gamma):
    """

    :param loss: The loss we want to use for our optimization process
    :param k: the latent dimension of our matrix factorization
    :param number_epochs: the number of times we want to go through all our training set during the training phase
    :param batch_size: the size of the batch to perform optimization algorithm
    :param l2_penal: ridge penalization (L2)
    :param gamma: our optimization learning rate
    :return: a factorization model ready to fit our input data
    """
    model = ExplicitFactorizationModel(loss=loss,
                                       embedding_dim=k,  # latent dimensionality
                                       n_iter=number_epochs,  # number of epochs of training
                                       batch_size=batch_size,  # minibatch size
                                       l2=l2_penal,  # strength of L2 regularization
                                       learning_rate=gamma)
    return model


def predict_output(model, test_interactions_object):
    """

    :param model: our trained model
    :param test_interactions_object: the interaction object on which we wish to predict ratings
    :return: desired output
    """
    y_predictions = model.predict(test_interactions_object.user_ids, test_interactions_object.item_ids)
    return np.round(y_predictions)


def create_output_df(y_predictions, test_df):
    """

    :param y_predictions: the predictions our model gave
    :param test_df: our test dataframe in the submission form
    :return: pandas dataframe with our predictions
    """
    test_df['rating'] = y_predictions
    test_df['rating'] = test_df['rating'].apply(lambda x: 1 if x < 1 else x)
    test_df['rating'] = test_df['rating'].apply(lambda x: 5 if x > 5 else x)
    return test_df


def create_submission_pd(test_df):
    """

    :param test_df: our final submission in pandas dataframe format
    :return: csv of our final submission
    """
    test_df.to_csv('data/final_submission.csv', index=False)

def preparing_data(raw_data):
    """

    :param raw_data: the dataframe loaded data
    :return: transforming data into a dataframe with users, movies and ratings in columns
    """
    # Splitting data in order to make use of it.
    splitted_data = splitting(raw_data)
    splitted_data['rating'] = raw_data['Prediction']
    splitted_data['rating'] = splitted_data['rating'].astype(float)
    return splitted_data

def train_model(model, X):
    """

    :param model: Our model
    :param X: the data on which we wish to train our model
    :return: our model trained
    """
    model.fit(X, verbose=True)
    return model