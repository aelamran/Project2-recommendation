import pandas as pd
import torch
from helpers_pj import *
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
torch.set_default_tensor_type('torch.DoubleTensor')

print("---------LOADING DATA-----------")
raw_data = pd.read_csv('data/data_train.csv')
splitted_data = splitting(raw_data)
splitted_data['rating'] = raw_data['Prediction']
splitted_data['rating'] = splitted_data['rating'].astype(float)

test_data = pd.read_csv('data/data_test.csv')
splitted_test_data = splitting(test_data)
splitted_test_data['rating'] = test_data['Prediction']
splitted_test_data['rating'] = splitted_test_data['rating'].astype(float)

explicit_interactions = Interactions(splitted_data['userid'].values,
                                     splitted_data['movieid'].values,
                                     splitted_data['rating'].values)

explicit_interactions_test = Interactions(splitted_test_data['userid'].values,
                                          splitted_test_data['movieid'].values,
                                          splitted_test_data['rating'].values)

model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=128,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3)

# train, test = random_train_test_split(explicit_interactions, random_state=np.random.RandomState(42))
# print('Split into \n {} and \n {}.'.format(train, test))

model.fit(explicit_interactions, verbose=True)
output = model.predict(explicit_interactions_test.user_ids, explicit_interactions_test.item_ids)
output = np.round(output)
create_csv_submission('data/data_test.csv', 'data/sub_spot.csv', output)
# train_rmse = rmse_score(model, train)
# test_rmse = rmse_score(model, test)

# print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))
