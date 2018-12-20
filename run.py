from MF_spotlight import *

LOSS = 'regression'  # Our chosen loss
K = 175  # Latent dimension of our matrix factorization
NB_EPOCHS = 10  # Number of times we go through our training set
BATCH_SIZE = 2400  # The batch size of our optimization algorithm
L2 = 1e-8  # Our lambda ridge penalization
GAMMA = 1e-3  # Our optimization learning rate

# Loading train data
print("---------LOADING DATA-----------")
raw_data_train = pd.read_csv('data/data_train.csv')
raw_data_output = pd.read_csv('data/data_test.csv')

print("---------PREPROCESSING DATA-----------")
df_input = preparing_data(raw_data_train)
df_output = preparing_data(raw_data_output)
input_interaction = create_input_for_spotlight(df_input['userid'].values,
                                               df_input['movieid'].values,
                                               df_input['rating'].values)

output_interaction = create_input_for_spotlight(df_output['userid'].values,
                                                df_output['movieid'].values,
                                                df_output['rating'].values)

print("---------TRAINING THE MODEL-----------")
model = create_model(loss=LOSS, k=K, number_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, l2_penal=L2, gamma=GAMMA)
model = train_model(model, input_interaction)
print("---------CREATING THE SUBMISSION-----------")
df_submission = create_output_df(y_predictions=predict_output(model, output_interaction), test_df=df_output)
create_submission_pd(df_submission)
