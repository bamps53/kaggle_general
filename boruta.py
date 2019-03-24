from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
 
# model
model = RandomForestRegressor(
    n_estimators=50
    , criterion='mse'
    , max_depth = 7
    , max_features = 'sqrt'
    , n_jobs=-1
    , verbose=True
    )
 
feat_selector = BorutaPy(model, 
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2, # 0: no output,1: displays iteration number,2: which features have been selected already
                         alpha=0.05, # 有意水準
                         max_iter=50, # 試行回数
                         random_state=1
                        )
 
feat_selector.fit(X_train.values, y_train.values)

[col for col in X_train.columns[feat_selector.support_]]
