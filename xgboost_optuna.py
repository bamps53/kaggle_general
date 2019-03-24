import optuna
 
def objective(trial):
    # hypyer param
    max_depth = trial.suggest_int('max_depth', 3, 7) 
    n_estimators = trial.suggest_loguniform('n_estimators', 10, 1000) 
    min_child_weight=trial.suggest_int('min_child_weight',0,20)
    eta=trial.suggest_loguniform('eta', 0.003, 0.3)
    
    xgb_params = {
        'eval_metric': 'rmse',
        'seed': 1337,
        'eta': eta,
        'n_estimators':n_estimators,
        'max_depth':max_depth,
        'subsample': 0.7,
        'min_child_weight': min_child_weight,
        'tree_method': 'gpu_hist',
        'device': 'gpu',
        'silent': 1,
    }
 
    # callback
    #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-merror')
 
    # model
    score=run_xgb_optuna(xgb_params, X_train, X_test ,y_train)

    return score
    
    import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

def run_xgb_optuna(params, X_train, X_test ,y_train):
    n_splits = 5
    verbose_eval = 100
    num_rounds = 60000
    early_stop = 200

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    cv_score=0

    i = 0

    for trn_idx, val_idx in kf.split(X_train, y_train):

        X_trn = X_train.iloc[trn_idx, :]
        X_val = X_train.iloc[val_idx, :]

        y_trn = y_train.iloc[trn_idx]
        y_val = y_train.iloc[val_idx]
        
        X_trn,X_val=rescue_encoding(X_trn,X_val)

        d_train = xgb.DMatrix(data=X_trn, label=y_trn.values, feature_names=X_trn.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val.values, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)
        

        cv_score+=model.best_score/n_splits
        
        i += 1
        
    return cv_score
    

 
study = optuna.create_study()
 
# https://optuna.readthedocs.io/en/stable/reference/study.html#optuna.study.Study.optimize
study.optimize(func=objective, # 実行する関数
               n_trials=30, # 試行回数
               timeout=None, # 与えられた秒数後に学習を中止します。default=None
               n_jobs=-1 # 並列実行するjob数
              )
              
print('best_param:{}'.format(study.best_params))
print('====================')
 
#最適化後の目的関数値
print('best_value:{}'.format(study.best_value))
print('====================')
 
#最適な試行
print('best_trial:{}'.format(study.best_trial))
print('====================')
 
# トライアルごとの結果を確認
for i in study.trials:
    print('param:{0}, eval_value:{1}'.format(i[5], i[2]))
print('====================')

result=study.trials_dataframe()
result.sort_values(by='value')

print(study.best_params)
