import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# 合并训练数据和测试数据
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 填补 MSZoning 缺失值
a = all_data.groupby(by=['MSSubClass', 'Street'])['MSZoning'].value_counts().reset_index(name='count')
all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = all_data.loc[all_data.MSZoning.isnull(), :].apply(lambda x: a.loc[(x.MSSubClass == a.MSSubClass) & (x.Street == a.Street), 'MSZoning'].values[0], axis=1)

# 填补 Utilities 缺失值
a = all_data.groupby(by=['MSSubClass'])['Utilities'].value_counts().reset_index(name='count')
all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = all_data.loc[all_data.Utilities.isnull()].apply(lambda x: a.loc[(x.MSSubClass == a.MSSubClass), 'Utilities'].values[0], axis=1)

##################### 构建随机森林预测LotFrontage
lf_df = all_data[['LotFrontage', 'MSSubClass', 'MSZoning', 'Street', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'OverallCond']]
lf_df_test = lf_df[lf_df.LotFrontage.isnull()]
lf_df_train = lf_df[lf_df.LotFrontage.notnull()]

lf_df_train = pd.get_dummies(lf_df_train)
lf_df_test = pd.get_dummies(lf_df_test)

target_lf = lf_df_train['LotFrontage']   # 训练目标

from sklearn.ensemble import RandomForestRegressor
train_lf = lf_df_train.drop('LotFrontage', axis=1)
test_lf = lf_df_test.drop('LotFrontage', axis=1)

train_sub = list(set(train_lf.columns) - set(test_lf.columns))
test_sub = set(test_lf.columns) - set(train_lf.columns)

train_lf.drop(train_sub, axis=1, inplace=True)
test_lf.drop(test_sub, axis=1, inplace=True)

rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=6)
rfr.fit(train_lf, target_lf)
target_lf = target_lf.values

scores = cross_val_score(estimator=rfr, X=train_lf, y=target_lf, cv=10, scoring='neg_mean_squared_error')

# features = pd.DataFrame()
# features['feature'] = train_lf.columns
# features['importance'] = rfr.feature_importances_
# features.sort_values('importance', inplace=True)
# features.set_index('feature', inplace=True)
# features.plot(kind='barh', figsize=(40, 40))
# plt.show()

# 现在让我们将我们的训练集和测试集转换为更紧凑的数据集。


model = SelectFromModel(rfr, prefit=True)
train_lf_reduced = model.transform(train_lf)

test_lf_reduced = model.transform(test_lf)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rfr_classifier = RandomForestRegressor(max_features='sqrt', random_state=10)
rfr_classifier2 = RandomForestRegressor(max_features='sqrt', random_state=10, n_estimators=50)


pipe = Pipeline([
    # ('std_scaler', StandardScaler()),
    ('clf', RandomForestRegressor(max_features='sqrt', random_state=10))
])

param_grid = {
    'clf__n_estimators': list(range(20, 50, 2)),
    'clf__max_depth': list(range(3, 60, 3))
}

gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
gsearch.fit(train_lf_reduced, target_lf)
ic(gsearch.best_params_, gsearch.best_score_)

# scores = cross_val_score(estimator=rfr_classifier2, X=train_lf_reduced, y=target_lf, cv=10, scoring='neg_mean_squared_error')
# ic(scores)


