import optuna
 
def objective(trial):
    # hypyer param
    num_leaves = trial.suggest_int('num_leaves', 15, 63) 
    max_depth = trial.suggest_int('max_depth', 5, 10) 
    learning_rate = 0.02516799521809343
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.4, 0.8) 
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.4, 0.8)
    min_child_weight = trial.suggest_uniform('min_child_weight', 0, 1) 
    num_rounds = 2000
    min_child_samples=trial.suggest_int('min_child_samplees',0,100)
    

    
    lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_bin' : 8,
              'num_leaves': num_leaves,
              'max_depth': max_depth,
              'learning_rate': learning_rate,
              'bagging_fraction': bagging_fraction,
              'feature_fraction': feature_fraction,
              'min_split_gain': 0.01,
              'min_child_samples': min_child_samples,
              'min_child_weight': min_child_weight,
              'data_random_seed': 123,
              'verbosity': -1,
              'early_stopping_rounds': 200,
              'num_rounds': num_rounds
}
 
    # callback
    #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-merror')
 
    # model
    score=run_lgb_reg_optuna(lgb_params, X_train, X_test ,y_train)

    return score
    
def run_lgb_reg_optuna(params, X_train, X_test ,y_train):
    n_splits = 5
    verbose_eval = 100
    num_rounds = 60000
    early_stop = 200

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    train_score=0
    cv_score=0

    i = 0

    for trn_idx, val_idx in kf.split(X_train, y_train):

        X_trn = X_train.iloc[trn_idx, :]
        X_val = X_train.iloc[val_idx, :]

        y_trn = y_train.iloc[trn_idx]
        y_val = y_train.iloc[val_idx]
        
        X_trn,X_val=rescue_encoding(X_trn,X_val)

        d_train = lgb.Dataset(data=X_trn, label=y_trn.values)
        d_valid = lgb.Dataset(data=X_val, label=y_val.values)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = lgb.train(train_set=d_train, valid_sets=d_valid, num_boost_round=num_rounds,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        trn_pred = model.predict(X_trn)
        valid_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        oof_train[val_idx] = valid_pred
        oof_test[:, i] = test_pred
        train_score+=mean_squared_error(y_trn,trn_pred)**0.5/n_splits
        cv_score+=model.best_score.get('valid_0')['rmse']/n_splits
        
        i += 1
    
    print('cv score: {}'.format(cv_score))
        
    return cv_score
    
    study = optuna.create_study()
 
# https://optuna.readthedocs.io/en/stable/reference/study.html#optuna.study.Study.optimize
study.optimize(func=objective, # 実行する関数
               n_trials=100, # 試行回数
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
