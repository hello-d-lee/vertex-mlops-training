import os
from pathlib import Path
import argparse
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

def get_args():
    """
    Get arguments from command line.
    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        required=False,
        default=os.getenv('AIP_TRAINING_DATA_URI'),
        type=str,
        help='path to read data')
    parser.add_argument(
        '--learning_rate',
        required=False,
        default=0.01,
        type=int,
        help='number of epochs')
    parser.add_argument(
        '--model_dir',
        required=False,
        default=os.getenv('AIP_MODEL_DIR'),
        type=str,
        help='dir to store saved model')
    parser.add_argument(
        '--config_path',
        required=False,
        default='../config.yaml',
        type=str,
        help='path to read config file')
    args = parser.parse_args()
    return args


def ingest_data(data_path, data_model_params):
    """
    Ingest data
    Args:
        data_path: path to read data
        data_model_params: data model parameters
    Returns:
        df: dataframe
    """
    # read training data
    df = pd.read_csv(data_path, sep=',',
                     dtype={col: 'string' for col in data_model_params['categorical_features']})
    return df


def preprocess_data(df, data_model_params):
    """
    Preprocess data
    Args:
        df: dataframe
        data_model_params: data model parameters
    Returns:
        df: dataframe
    """

    # convert nan values because pd.NA ia not supported by SimpleImputer
    # bug in sklearn 0.23.1 version: https://github.com/scikit-learn/scikit-learn/pull/17526
    # decided to skip NAN values for now
    df.replace({pd.NA: np.nan}, inplace=True)
    df.dropna(inplace=True)

    # get features and labels
    x = df[data_model_params['numerical_features'] + data_model_params['categorical_features'] + [
        data_model_params['weight_feature']]]
    y = df[data_model_params['target']]

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=data_model_params['train_test_split']['test_size'],
                                                        random_state=data_model_params['train_test_split'][
                                                            'random_state'])
    return x_train, x_test, y_train, y_test


def build_pipeline(learning_rate, model_params):
    """
    Build pipeline
    Args:
        learning_rate: learning rate
        model_params: model parameters
    Returns:
        pipeline: pipeline
    """
    # build pipeline
    pipeline = Pipeline([
        # ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('model', xgb.XGBClassifier(learning_rate=learning_rate,
                                    use_label_encoder=False, #deprecated and breaks Vertex AI predictions
                                    **model_params))
    ])
    return pipeline


def main():
    print('Starting training...')
    args = get_args()
    data_path = args.data_path
    learning_rate = args.learning_rate
    model_dir = args.model_dir
    config_path = args.config_path

    # read config file
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    data_model_params = config['data_model_params']
    model_params = config['model_params']

    # ingest data
    print('Reading data...')
    data_df = ingest_data(data_path, data_model_params)

    # preprocess data
    print('Preprocessing data...')
    x_train, x_test, y_train, y_test = preprocess_data(data_df, data_model_params)
    sample_weight = x_train.pop(data_model_params['weight_feature'])
    sample_weight_eval_set = x_test.pop(data_model_params['weight_feature'])

    # train lgb model
    print('Training model...')
    xgb_pipeline = build_pipeline(learning_rate, model_params)
    # need to use fit_transform to get the encoded eval data
    x_train_transformed = xgb_pipeline[:-1].fit_transform(x_train)
    x_test_transformed = xgb_pipeline[:-1].transform(x_test)
    xgb_pipeline[-1].fit(x_train_transformed, y_train,
                         sample_weight=sample_weight,
                         eval_set=[(x_test_transformed, y_test)],
                         sample_weight_eval_set=[sample_weight_eval_set],
                         eval_metric='error',
                         early_stopping_rounds=50,
                         verbose=True)
    # save model
    print('Saving model...')
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_pipeline, f'{model_dir}/model.joblib')


if __name__ == "__main__":
    main()
