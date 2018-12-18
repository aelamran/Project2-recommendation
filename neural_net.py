from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
from helpers_pj import *
import matplotlib.pyplot as plt
import os
import zipfile
import requests
import tqdm

import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras

print("------------PREPROCESSING DATA------------")
raw_data = pd.read_csv('data/data_train.csv')
ratings = splitting(raw_data)
ratings['rating'] = raw_data['Prediction']


# Also, make vectors of all the movie ids and user ids. These are
# pandas categorical data, so they range from 1 to n_movies and 1 to n_users, respectively.
movieid = ratings.movieid.values
userid = ratings.userid.values

y = np.zeros((ratings.shape[0], 5))
y[np.arange(ratings.shape[0]), ratings.rating - 1] = 1

# Now, the deep learning classifier

# First, we take the movie and vectorize it.
# The embedding layer is normally used for sequences (think, sequences of words)
# so we need to flatten it out.
# The dropout layer is also important in preventing overfitting
movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(1000 + 1, 32)(movie_input))
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

# Same thing for the users
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(10000 + 1, 32)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

# Next, we join them all together and put them
# through a pretty standard deep learning architecture
input_vecs = keras.layers.concatenate([movie_vec, user_vec], axis=-1)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(128, activation='relu')(nn)

# Finally, we pull out the result!
result = keras.layers.Dense(5, activation='softmax')(nn)

# And make a model from it that we can actually run.
model = kmodels.Model([movie_input, user_input], result)
model.compile('adam', 'categorical_crossentropy')

# If we wanted to inspect part of the model, for example, to look
# at the movie vectors, here's how to do it. You don't need to
# compile these models unless you're going to train them.
final_layer = kmodels.Model([movie_input, user_input], nn)
movie_vec = kmodels.Model(movie_input, movie_vec)


# Split the data into train and test sets...
a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = train_test_split(movieid, userid, y)

# And of _course_ we need to make sure we're improving, so we find the MAE before
# training at all.
print(mean_absolute_error(np.argmax(b_y, 1)+1, np.argmax(model.predict([b_movieid, b_userid]), 1)+1))
try:
    history = model.fit([a_movieid, a_userid], a_y,
                         nb_epoch=10,
                         validation_data=([b_movieid, b_userid], b_y))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
except KeyboardInterrupt:
    pass

# This is the number that matters. It's the held out
# test set score. Note the + 1, because np.argmax will
# go from 0 to 4, while our ratings go 1 to 5.
print(mean_absolute_error(np.argmax(b_y, 1)+1, np.argmax(model.predict([b_movieid, b_userid]), 1)+1))