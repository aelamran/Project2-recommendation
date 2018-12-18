import pandas as pd
from helpers_pj import *
from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel



print("---------PREPRO WITH LAB 10-----------")
raw_data_lab10 = load_data('data/data_train.csv')
user_ids, item_ids = raw_data_lab10.nonzero()
train, test = split_data(raw_data_lab10, split=0.1)

print(train)
print(test)

# print("------------PREPROCESSING DATA------------")
# raw_data = pd.read_csv('data/data_train.csv')
# split_user_movie = splitting(raw_data)
# split_user_movie['eval'] = raw_data['Prediction']
# rating_table = split_user_movie.pivot(index='user_ids', columns='item_ids', values='eval')
#
# train, test = split_data(raw_data)
#
#
#
#
#
#
#
implicit_interactions = Interactions(user_ids, item_ids)
# explicit_interactions = Interactions(split_user_movie['user_ids'].values,
#                                      split_user_movie['item_ids'].values,
#                                      split_user_movie['eval'].values)

implicit_model = ImplicitFactorizationModel()
implicit_model.fit(implicit_interactions)
implicit_model.predict(user_ids, item_ids=None)

