import xgb_model as x
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import numerapi

napi = numerapi.NumerAPI()
current_rnd = napi.get_current_round()

path_server = f'/home/mike/Documents/numerai/numerai_data/numerai_dataset_{str(current_rnd)}/'

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

def test_model_xgb(train, tourn, feature_names, **kwargs):
    """ Train xgboost model and return train and tourn datasets """
    
    model = XGBRegressor(tree_method='gpu_hist', **kwargs)
    print(f'{x.get_time()} Training model...')
    print(model)
    model.fit(train[feature_names], train[TARGET_NAME])
    print(f'{x.get_time()} Done.')

    # Generate predictions on both training and tournament data
    print(f'{x.get_time()} Generating predictions...', end='', flush=True)
    train[PREDICTION_NAME] = model.predict(train[feature_names])
    tourn[PREDICTION_NAME] = model.predict(tourn[feature_names])
    print(f'{x.get_time()} Done.')
    return train, tourn, model
    
def diagnostics(train, tourn, feature_names):
    print(f'{x.get_time()} Calculating diagnostics...', end='', flush=True)
    
    diagnostics = {}
    # Correlation, std, sharpe
    valid = tourn[tourn.data_type == 'validation']
    valid_corr = valid.groupby("era").apply(x.score)
    valid_corr_mean = valid_corr.mean()
    valid_corr_std = valid_corr.std(ddof=0)
    valid_sharpe = valid_corr_mean / valid_corr_std
    
    # Drawdown
    rolling_max = (valid_corr + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (valid_corr + 1).cumprod()
    max_drawdown = -(rolling_max - daily_value).max()
    
    # Feature exposure
    feature_exposures = valid[feature_names].apply(lambda d: x.correlation(valid[PREDICTION_NAME], d), axis=0)
    max_per_era = valid.groupby("era").apply(lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
    max_feature_exposure = max_per_era.mean()
    
    # Feature neutral mean
    feature_neutral_mean = x.get_feature_neutral_mean(valid)
    
    # MMC diagnostics from example predictions
    example_preds = pd.read_csv(f'{path_server}example_predictions.csv').set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[valid.index]
    valid["ExamplePreds"] = validation_example_preds
    mmc_scores = []
    corr_scores = []
    for _, y in valid.groupby("era"):
        series = x.neutralize_series(pd.Series(x.unif(y[PREDICTION_NAME])),
                                   pd.Series(x.unif(y["ExamplePreds"])))
        mmc_scores.append(np.cov(series, y[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(x.correlation(x.unif(y[PREDICTION_NAME]), y[TARGET_NAME]))

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
    per_era_corrs = full_df.groupby('era').apply(lambda d: x.correlation(x.unif(d["prediction"]), x.unif(d["example_preds"])))
    corr_with_example_preds = per_era_corrs.mean()
    
    diagnostic_names = [
                        'validation_corr_mean', 'validation_corr_std', 'validation_sharpe', 'max_drawdown', 'max_feature_exposure',
                        'feature_neutral_mean', 'validation_mmc_mean', 'validation_mmc_sharpe', 'corr_plus_mmc_mean', 'corr_plus_mmc_sharpe',
                        'corr_with_example_preds'
                        ]
    
    diagnostic_values = [
                        valid_corr_mean, valid_corr_std, valid_sharpe, max_drawdown, max_feature_exposure,
                        feature_neutral_mean, val_mmc_mean, val_mmc_sharpe, corr_plus_mmc_mean, corr_plus_mmc_sharpe,
                        corr_with_example_preds
                        ]
    for i in range(len(diagnostic_names)):
        diagnostics[diagnostic_names[i]] = diagnostic_values[i]
        
    print(f'{x.get_time()} Done.')
    return diagnostics
    
def update_diagnostics(model, diagnostics):
    # existing_file = pd.read_csv(filename, index_col=0)
    params = pd.Series(model.get_params())
    diagnostics = pd.Series(diagnostics)
    params_diagnostics = pd.concat([params, diagnostics], axis=1)
    params_diagnostics.to_csv('params_diagnostics.csv')