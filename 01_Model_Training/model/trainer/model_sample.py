# Import python modules
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras_tuner 
from google.cloud import aiplatform

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy
import pandas
import json, os

# Declare variables
REGION = "us-central1"
PROJECT_ID = "uki-mlops-dev-demo"
MODEL_PATH='gs://'+PROJECT_ID+'-bucket/model/'
DATASET_PATH='gs://'+PROJECT_ID+'-bucket/area_cover_dataset.csv'
PIPELINE_ROOT = 'gs://'+PROJECT_ID+'-bucket'
MODEL_ARTIFACTS_LOCATION ='gs://'+PROJECT_ID+'-bucket/'

# Read the area_cover_dataset csv data into pandas dataframe
area_cover_dataframe = pandas.read_csv(DATASET_PATH)

# Function that takes the area cover dataframe and converts the two categorical (string) columns into indexed values
def index(dataframe):
    for col in dataframe.columns:
        if col=="Wilderness_Area":
            test1_column = dataframe['Wilderness_Area']
            test1_index = pandas.Categorical(test1_column)
            dataframe['Wilderness_Area'] = test1_index.codes
        elif col=="Soil_Type":
            test2_column = dataframe['Soil_Type']
            test2_index = pandas.Categorical(test2_column)
            dataframe['Soil_Type'] = test2_index.codes
        else:
            print("non cat col")
    return dataframe

# Extract the feature columns into a new dataframe called scaler_features that has been standardized using the sklearn.preprocessing.StandardScaler method.
# The features are all columns from the area cover dataset except the "Area_Cover" column
indexed_dataframe = index(area_cover_dataframe)
features_dataframe = indexed_dataframe.drop("Area_Cover", axis = 1)
standard_scaler = StandardScaler()
df = standard_scaler.fit_transform(features_dataframe)
scaled_features = pandas.DataFrame(df, columns=features_dataframe.columns)

# Create a binary matrix containing the categorical Area_Cover column data converted using keras.utils.to_categorical()
labels_dataframe = indexed_dataframe["Area_Cover"]
categorical_labels = to_categorical(labels_dataframe)

# Split the dataset into model training and validation data
dfx_train, dfx_val, dfy_train, dfy_val = train_test_split(scaled_features.values, categorical_labels, test_size=0.2)

# Create a function that returns a sequential categorical model function with a hyperparameter tuning layer
def build_model(hptune):
    model = Sequential()
    model.add(Dense(128, input_shape = (12,), activation = "relu"))
    # add hyperparmeter tuning layer
    model.add(Dense(units=hptune.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    # add output layer w 7 label classes
    model.add(Dense(units=7, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create a Keras Hyperband Hyperparameter tuner with an accuracy objective

tuner =  keras_tuner.Hyperband(build_model, objective='val_accuracy', max_epochs=50)

# Define an early stopping callback using that stops when the validation loss quantity does not improve after 5 epochs
stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Perform a Keras Tuner Search for the best hyperparameter configurations using the training data split over 50 epochs
tuner.search(dfx_train, dfy_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters for the model as determined from the search
best_hyperparameters=tuner.get_best_hyperparameters(num_trials=10)[0]

# Create a new model using the best_hyperparameters and train it. 
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(dfx_train, dfy_train, epochs=50, validation_split=0.2)

# Using the model training history find and print out the epoch with the best validation accuracy. 
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Print out the Model test loss and test accuracy by evaluating the validation data split. 
eval_result = model.evaluate(dfx_val, dfy_val)
print("[Model test loss, test accuracy]:", eval_result)

# Create a new model (hypermodel) using the best_hyperparameters and retrain. 
hypermodel = tuner.hypermodel.build(best_hyperparameters)
# Retrain the model using the number of epochs that was previously determined to be the best. 
hypermodel.fit(dfx_train, dfy_train, epochs=best_epoch, validation_split=0.2)

# Print out the test loss and test accuracy for hypermodel by evaluating the validation data split. 
eval_result = hypermodel.evaluate(dfx_val, dfy_val)
print("[Hypermodel test loss, test accuracy]:", eval_result)

# Save the hypertuned model
# NB the MODEL_PATH bucket must be created before this will succeed and it must be in the same location as the model.
# e.g. gsutil mb -l us-central1  gs://${PROJECT_ID}-bucket
hypermodel.save(MODEL_PATH)