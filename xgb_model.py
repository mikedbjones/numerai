#!/usr/bin/env python

import csv
from pathlib import Path
import os
import json

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
from xgboost import XGBRegressor
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from datetime import datetime
import numerapi
import pyinputplus as pyip

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

MODEL_FILE = Path("example_model.xgb")

# keys to be used in API calls for submission
PUBLIC_ID = os.environ.get('PUBLIC_ID')
SECRET_KEY = os.environ.get('SECRET_KEY')

# call API without keys for general use
napi = numerapi.NumerAPI()
current_rnd = napi.get_current_round()

path_server = f'/home/mike/Documents/numerai/numerai_data/numerai_dataset_{str(current_rnd)}/'
path_linux = f'/mnt/c/Users/Mike Jones/Documents/numerai_data/numerai_dataset_{str(current_rnd)}/'
path_windows = f'C:\\Users\\Mike Jones\\Documents\\numerai_data\\numerai_dataset_{str(current_rnd)}\\'

''' Functions for metrics '''


# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        # print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: correlation(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)


def diagnostics(train, tourn, feature_names):
    print(f'{get_time()} Calculating diagnostics...', end='', flush=True)

    diag = {}
    # Correlation, std, sharpe
    valid = tourn[tourn.data_type == 'validation']
    valid_corr = valid.groupby("era").apply(score)
    valid_corr_mean = valid_corr.mean()
    valid_corr_std = valid_corr.std(ddof=0)
    valid_sharpe = valid_corr_mean / valid_corr_std

    # Drawdown
    rolling_max = (valid_corr + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (valid_corr + 1).cumprod()
    max_drawdown = -(rolling_max - daily_value).max()

    # Feature exposure
    feature_exposures = valid[feature_names].apply(lambda d: correlation(valid[PREDICTION_NAME], d), axis=0)
    max_per_era = valid.groupby("era").apply(lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
    max_feature_exposure = max_per_era.mean()

    # Feature neutral mean
    feature_neutral_mean = get_feature_neutral_mean(valid)

    # MMC diagnostics from example predictions
    example_preds = pd.read_csv(f'{path_server}example_predictions.csv').set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[valid.index]
    valid["ExamplePreds"] = validation_example_preds
    mmc_scores = []
    corr_scores = []
    for _, y in valid.groupby("era"):
        series = neutralize_series(pd.Series(unif(y[PREDICTION_NAME])),
                                   pd.Series(unif(y["ExamplePreds"])))
        mmc_scores.append(np.cov(series, y[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(correlation(unif(y[PREDICTION_NAME]), y[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - valid_sharpe

    # Check correlation with example predictions
    full_df = pd.concat([validation_example_preds, valid[PREDICTION_NAME], valid["era"]], axis=1)
    full_df.columns = ["example_preds", "prediction", "era"]
    per_era_corrs = full_df.groupby('era').apply(lambda d: correlation(unif(d["prediction"]), unif(d["example_preds"])))
    corr_with_example_preds = per_era_corrs.mean()

    diagnostic_names = [
        'validation_corr_mean', 'validation_corr_std', 'validation_sharpe', 'max_drawdown', 'max_feature_exposure',
        'feature_neutral_mean', 'validation_mmc_mean', 'validation_mmc_sharpe', 'corr_plus_mmc_mean',
        'corr_plus_mmc_sharpe',
        'corr_with_example_preds'
    ]

    diagnostic_values = [
        valid_corr_mean, valid_corr_std, valid_sharpe, max_drawdown, max_feature_exposure,
        feature_neutral_mean, val_mmc_mean, val_mmc_sharpe, corr_plus_mmc_mean, corr_plus_mmc_sharpe,
        corr_with_example_preds
    ]
    for i in range(len(diagnostic_names)):
        diag[diagnostic_names[i]] = diagnostic_values[i]

    print(f'{get_time()} Done.')
    return diag


def metrics_simple(train, tourn):
    print(f'{get_time()} Calculating metrics...')
    # Check the per-era correlations on the training set (in sample)
    train_correlations = train.groupby("era").apply(score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")

    """Validation Metrics"""
    # Check the per-era correlations on the validation set (out of sample)
    validation_data = tourn[tourn.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and "
          f"std {validation_correlations.std(ddof=0)}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    # Check the "sharpe" ratio on the validation set
    validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    print(f"Validation Sharpe: {validation_sharpe}")

    print(f'{get_time()} Done.')


def get_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


''' Functions for processing '''


def download_data():
    download = pyip.inputChoice(['y', 'n'], prompt='Download current data? y/n...')
    if download == 'y' or download == 'Y':
        print(f'{get_time()} Downloading...', end='', flush=True)
        current_data = napi.download_current_dataset(dest_path='/home/mike/Documents/numerai/numerai_data')
        #        current_data = napi.download_current_dataset(dest_path = '/mnt/c/Users/Mike Jones/Documents/numerai_data')
        #        current_data = napi.download_current_dataset(dest_path = f'C:\\Users\\Mike Jones\\Documents\\numerai_data')
        print(f'{get_time()} Done.')
    else:
        print(f'{get_time()} Not downloading.')


# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)

    # Memory constrained? Try this instead (slower, but more memory efficient)
    # see https://forum.numer.ai/t/saving-memory-with-uint8-features/254
    # dtypes = {f"target": np.float16}
    # to_uint8 = lambda x: np.uint8(float(x) * 4)
    # converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    # df = pd.read_csv(file_path, dtype=dtypes, converters=converters)

    return df


def load_data(path=path_server):
    print(f'{get_time()} Loading data from round {current_rnd}...', end='', flush=True)
    # The training data is used to train your model how to predict the targets.
    train = read_csv(f'{path}numerai_training_data.csv')
    # The tournament data is the data that Numerai uses to evaluate your model.
    tourn = read_csv(f'{path}numerai_tournament_data.csv')
    print(f'{get_time()} Done.')
    return train, tourn


def load_features(df):
    feature_names = [
        f for f in df.columns if f.startswith("feature")
    ]
    print(f"{get_time()} Loaded {len(feature_names)} features.")
    return feature_names


def get_group_stats(df):
    print(f'{get_time()} Calculating group stats...', end='', flush=True)
    for group in ["intelligence", "charisma", "strength", "dexterity", "constitution", "wisdom"]:
        cols = [col for col in df.columns if group in col]
        df[f"feature_{group}_group_mean"] = df[cols].mean(axis=1)
        df[f"feature_{group}_group_std"] = df[cols].std(axis=1)
        df[f"feature_{group}_group_skew"] = df[cols].skew(axis=1)
    print(f'{get_time()} Done.')
    return df


def get_feature_groups(df):
    feature_intel = [f for f in df.columns if 'intelligence' in f]
    feature_charisma = [f for f in df.columns if 'charisma' in f]
    feature_strength = [f for f in df.columns if 'strength' in f]
    feature_dexterity = [f for f in df.columns if 'dexterity' in f]
    feature_constitution = [f for f in df.columns if 'constitution' in f]
    feature_wisdom = [f for f in df.columns if 'wisdom' in f]

    feature_all = [feature_intel, feature_charisma, feature_strength, feature_dexterity,
                   feature_constitution, feature_wisdom]
    return feature_all


def feature_interactions(df, features):
    interactions = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    interactions.fit(df[features])
    col_names = interactions.get_feature_names(features)

    df_interact = pd.DataFrame(interactions.transform(df[features]), columns=col_names, index=df.index)
    df_interact = df_interact.astype(np.float16)  # float16 for memory
    df_interact = df_interact.drop(columns=features)  # drop original features from df_interact
    df = pd.concat([df, df_interact], axis=1)
    return df


def feature_interactions_intel_dexte(df):
    print(f'{get_time()} Adding 2nd order interactions between intelligence and dexterity features...', end='',
          flush=True)
    feature_all = get_feature_groups(df)

    features = feature_all[0] + feature_all[3]  # intelligence+dexterity
    df = feature_interactions(df, features)
    print(f'{get_time()} Done.')
    return df


def run_model_xgb(train, tourn, feature_names, numerai_model_name, model_round):
    if model_round == '':
        model_round = f'models/{numerai_model_name}/round{current_rnd}.xgb'
    
    MODEL_FILE = Path(model_round)
    
    with open(f'models/{numerai_model_name}/current_parameters.json', 'r') as fp:
        current_parameters = json.loads(fp.read())

    model = XGBRegressor(**current_parameters)
    print(f'{get_time()} Model:\n{model}')

    # delete below line when checked things work OK
    if model_round != '':
        if MODEL_FILE.is_file():
            print(f'{get_time()} Loading pre-trained model from {MODEL_FILE}...', end='', flush=True)
            model.load_model(MODEL_FILE)
            print(f'{get_time()} Done.')
        else:
            print(f'{get_time()} Training model...', end='', flush=True)
            model.fit(train[feature_names], train[TARGET_NAME])
            print(f'{get_time()} Done.')
            print(f'{get_time()} Saving model to {model_round}...', end='', flush=True)
            model.save_model(MODEL_FILE)
            print(f'{get_time()} Done.')
    #else:
    #    print(f'{get_time()} Training model...', end='', flush=True)
    #    model.fit(train[feature_names], train[TARGET_NAME])
    #    print(f'{get_time()} Done.')

    # Generate predictions on both training and tournament data
    print(f'{get_time()} Generating predictions...', end='', flush=True)
    train[PREDICTION_NAME] = model.predict(train[feature_names])
    tourn[PREDICTION_NAME] = model.predict(tourn[feature_names])

    print(f'{get_time()} Done.')
    return train, tourn, model_round


def _neutralize(df, target=PREDICTION_NAME, by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()


def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)


def normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]


def neut_by_era(df, feature_names, prop=1):
    print(f'{get_time()} Neutralizing {100 * prop}% by era...', end='', flush=True)
    df["preds_neutralized"] = df.groupby("era").apply(
        lambda x: normalize_and_neutralize(x, [PREDICTION_NAME], feature_names,
                                           prop))  # neutralize by prop% within each era)
    scaler = MinMaxScaler()
    df["preds_neutralized"] = scaler.fit_transform(df[["preds_neutralized"]])  # transform back to 0-1
    df[PREDICTION_NAME] = df['preds_neutralized']
    df = df.drop(columns=['preds_neutralized'])
    print(f'{get_time()} Done.')
    return df


def submit(tourn, numerai_model_name, path=path_server):
    print(f'{get_time()} Exporting to CSV...', end='', flush=True)
    # Save predictions as a CSV and upload to https://numer.ai
    tourn[PREDICTION_NAME].to_csv(f'{path}submission_{numerai_model_name}.csv', header=True)
    print(f'{get_time()} Done.')


def upload_predictions(numerai_model_name, path=path_server):
    # call API with keys for uploading
    napi = numerapi.NumerAPI(PUBLIC_ID, SECRET_KEY)
    model_id = napi.get_models()[f'{numerai_model_name}']

    submit = pyip.inputChoice(['y', 'n'], prompt='Submit to Numerai? y/n...')
    if submit == 'y' or submit == 'Y':
        print(f'{get_time()} Submitting...', end='', flush=True)
        napi.upload_predictions(f'{path}submission_{numerai_model_name}.csv', model_id=model_id)
        print(f'{get_time()} Done.')
    else:
        print(f'{get_time()} Not submitting.')

        # check submission status
    #status = napi.submission_status(model_id)
    #print(f'Submission status:\n{status}')


''' Main '''


def main():
    print(f'***Round number {current_rnd}***')
    download_data()
    train, tourn = load_data()
    feature_names = load_features(train)
    train = feature_interactions_intel_dexte(train)
    tourn = feature_interactions_intel_dexte(tourn)
    feature_names = load_features(train)
    feature_names_orig = feature_names[:310]
    
    def run_numerai_model(train, tourn, feature_names, feature_names_orig, numerai_model_name, model_round):
        run = pyip.inputChoice(['y', 'n'], prompt=f'Run model {numerai_model_name}? y/n...')
        if run == 'y' or run == 'Y':
            print(f'{get_time()} Running model {numerai_model_name} ...')
            train, tourn, model_round = run_model_xgb(train, tourn, feature_names, numerai_model_name, model_round)
            train = neut_by_era(train, feature_names_orig, .5)
            tourn = neut_by_era(tourn, feature_names_orig, .5)
            diag = diagnostics(train, tourn, feature_names_orig)
            print(diag)
            submit(tourn, numerai_model_name)
            upload_predictions(numerai_model_name)
        else:
            print(f'{get_time()} Skipping model {numerai_model_name}.')
    
    run_numerai_model(train, tourn, feature_names, feature_names_orig, 'quantized', 'models/quantized/round257.xgb')
    run_numerai_model(train, tourn, feature_names, feature_names_orig, 'quantized2', 'models/quantized2/round264.xgb')


if __name__ == '__main__':
    main()
