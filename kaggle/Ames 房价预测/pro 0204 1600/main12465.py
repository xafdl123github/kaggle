import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

SubmitId = test_data['Id']

# 去掉异常值 2个
train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index, inplace=True)

train_data['SalePrice_Log'] = np.log(train_data['SalePrice'])

# 合并训练数据和测试数据
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 填补 MSZoning 缺失值
a = all_data.groupby(by=['MSSubClass', 'Street'])['MSZoning'].value_counts().reset_index(name='count')
all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = all_data.loc[all_data.MSZoning.isnull(), :].apply(lambda x: a.loc[(x.MSSubClass == a.MSSubClass) & (x.Street == a.Street), 'MSZoning'].values[0], axis=1)

# 填补 Utilities 缺失值
a = all_data.groupby(by=['MSSubClass'])['Utilities'].value_counts().reset_index(name='count')
all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = all_data.loc[all_data.Utilities.isnull()].apply(lambda x: a.loc[(x.MSSubClass == a.MSSubClass), 'Utilities'].values[0], axis=1)

# 均值法填补 LogFrontage [MSSubClass,OverallCond,BldgType,LotConfig,Neighborhood,MSZoning]
a = all_data[all_data.LotFrontage.notnull()].groupby(by=['MSSubClass', 'OverallCond', 'BldgType'])['LotFrontage'].mean()
a = a.reset_index()[['MSSubClass', 'OverallCond', 'BldgType', 'LotFrontage']]
def fill_lf(row):
    fill_val = a[(a.MSSubClass == row.MSSubClass) & (a.OverallCond == row.OverallCond) & (a.BldgType == row.BldgType)]['LotFrontage'].values
    if len(fill_val) == 0:
        return np.nan
    else:
        return fill_val[0]
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = all_data[all_data.LotFrontage.isnull()].apply(fill_lf, axis=1)

"""第二次"""
a = all_data[all_data.LotFrontage.notnull()].groupby(by=['MSSubClass'])['LotFrontage'].mean()
a = a.reset_index()[['MSSubClass', 'LotFrontage']]

def fill_lf2(row):
    fill_val = a[(a.MSSubClass == row.MSSubClass)]['LotFrontage'].values
    if len(fill_val) == 0:
        return np.nan
    else:
        return fill_val[0]
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = all_data[all_data.LotFrontage.isnull()].apply(fill_lf2, axis=1)

"""第三次"""
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = all_data.loc[all_data.LotFrontage.notnull(), 'LotFrontage'].mean()

# 采用众数法填补Exterior1st
all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'

# 采用众数法填补Exterior1st
all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'

# 填补 MasVnrType 缺失值
a = all_data.groupby(by=['BldgType', 'HouseStyle', 'OverallQual', 'MSSubClass'])['MasVnrType'].value_counts().reset_index(name='count')
all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType'] = all_data.loc[all_data.MasVnrType.isnull(), :].apply(lambda x: a.loc[(x.BldgType == a.BldgType) & (x.HouseStyle == a.HouseStyle) & (x.OverallQual == a.OverallQual) & (x.MSSubClass == a.MSSubClass), 'MasVnrType'].values[0], axis=1)

# 填补 MasVnrArea 缺失值
"""第一次"""
all_data.loc[all_data.MasVnrType == 'None', ['MasVnrArea']] = 0.0

"""第二次"""
a = all_data[(all_data.MasVnrArea.notnull()) & (all_data.MasVnrType != 'None')].groupby(by=['OverallQual'])['MasVnrArea'].mean().reset_index()[['OverallQual', 'MasVnrArea']]
def fill_mvr(row):
    t = a.loc[(a.OverallQual == row.OverallQual), 'MasVnrArea']
    return t.values[0]
all_data.loc[all_data.MasVnrArea.isnull(), 'MasVnrArea'] = all_data[all_data.MasVnrArea.isnull()].apply(fill_mvr, axis=1)

# 填补 BsmtQual 缺失值
def fill_BsmtQual(row):
    t = all_data.loc[(all_data.OverallCond == row.OverallCond), 'BsmtQual'].value_counts()
    return t.index[0]
all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual'] = all_data[all_data.BsmtQual.isnull()].apply(fill_BsmtQual, axis=1)

# 填补 BsmtCond 缺失值
def fill_BsmtCond(row):
    t = all_data.loc[(all_data.OverallCond == row.OverallCond), 'BsmtCond'].value_counts()
    return t.index[0]
all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond'] = all_data[all_data.BsmtCond.isnull()].apply(fill_BsmtCond, axis=1)

# 填补 BsmtExposure 缺失值
def fill_BsmtExposure(row):
    t = all_data.loc[(all_data.OverallCond == row.OverallCond), 'BsmtExposure'].value_counts()
    return t.index[0]
all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = all_data[all_data.BsmtExposure.isnull()].apply(fill_BsmtExposure, axis=1)

# 填补 BsmtFinType1 缺失值
def fill_BsmtFinType1(row):
    t = all_data.loc[(all_data.OverallCond == row.OverallCond), 'BsmtFinType1'].value_counts()
    return t.index[0]
all_data.loc[all_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = all_data[all_data.BsmtFinType1.isnull()].apply(fill_BsmtFinType1, axis=1)

# 填补 BsmtFinType2 缺失值
def fill_BsmtFinType2(row):
    t = all_data.loc[(all_data.OverallCond == row.OverallCond), 'BsmtFinType2'].value_counts()
    return t.index[0]
all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = all_data[all_data.BsmtFinType2.isnull()].apply(fill_BsmtFinType2, axis=1)

# 填补 BsmtFinSF1 缺失值
all_data.loc[all_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = 550

# 填补 BsmtFinSF2 缺失值
all_data.loc[all_data.BsmtFinSF2.isnull(), 'BsmtFinSF2'] = 0.0

# a = all_data[all_data.BsmtFinSF2.isnull()][['1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinType1', 'BsmtFinType2']]
# ic(a)
# c = all_data[all_data['BsmtFinType2'] == 'Unf'][['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF']]
# ic(c)
# ic(c.BsmtFinSF2.unique())
# ic(c.BsmtFinSF2.value_counts())

# 填补 TotalBsmtSF 缺失值
"""TotalBsmtSF(地下室总面积) = BsmtFinSF1(地下室装饰面积) + BsmtFinSF2(地下室装饰面积) + BsmtUnfSF(未装饰的地下室面积)"""

all_data.loc[all_data.TotalBsmtSF.isnull(), 'TotalBsmtSF'] = 896.0

# 填补 BsmtUnfSF 缺失值
t = all_data.loc[all_data.BsmtUnfSF.isnull(), 'TotalBsmtSF'] - all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtFinSF1'] -  all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtFinSF2']
'''获取标量值'''
filled_val = t.values[0]
'''填补'''
all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = filled_val

# 填补 Electrical 缺失值
all_data.loc[all_data.Electrical.isnull(), 'Electrical'] = 'SBrkr'

# 填补 BsmtFullBath 缺失值
'''第一个缺失值'''
all_data.loc[all_data.Id == 2189, 'BsmtFullBath'] = 0.0
'''第二个缺失值'''
all_data.loc[all_data.Id == 2121, 'BsmtFullBath'] = 1.0

# 填补 BsmtHalfBath 缺失值
'''第一个缺失值'''
all_data.loc[all_data.Id == 2189, 'BsmtHalfBath'] = 0.0
'''第二个缺失值'''
all_data.loc[all_data.Id == 2121, 'BsmtHalfBath'] = 0.0

# 填补 KitchenQual 缺失值
t = all_data.groupby('OverallCond')['KitchenQual'].value_counts().reset_index(name='count')

def fill_kq(row):
    a = t[t.OverallCond == row.OverallCond]['KitchenQual']
    return a.values[0]

all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual']=all_data[all_data.KitchenQual.isnull()].apply(fill_kq, axis=1)

# 填补 Functional 缺失值
t = all_data.groupby('OverallCond')['Functional'].value_counts().reset_index(name='count')

def fill_kq(row):
    a = t[t.OverallCond == row.OverallCond]['Functional']
    return a.values[0]

all_data.loc[all_data.Functional.isnull(), 'Functional']=all_data[all_data.Functional.isnull()].apply(fill_kq, axis=1)

# 填补 GarageCars 缺失值
t = all_data.groupby(['OverallQual', 'YearBuilt'])['GarageCars'].value_counts().reset_index(name='count')

def fill_gc(row):
    a = t[(t.OverallQual == row.OverallQual) & (t.YearBuilt == row.YearBuilt)]['GarageCars']
    return a.values[0]

all_data.loc[all_data.GarageCars.isnull(), 'GarageCars'] = all_data[all_data.GarageCars.isnull()].apply(fill_gc, axis=1)

# 填补 GarageArea 缺失值
t = all_data.groupby('GarageCars')['GarageArea'].value_counts().reset_index(name='count')

def fill_ga(row):
    a = t[t.GarageCars == row.GarageCars]['GarageArea']
    return a.values[0]

all_data.loc[all_data.GarageArea.isnull(), 'GarageArea']=all_data[all_data.GarageArea.isnull()].apply(fill_ga, axis=1)

# 通过随机森林填补 GarageQual 缺失值
init_f = ['MSSubClass','MSZoning','OverallCond','YearBuilt','YearRemodAdd','Foundation','BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','1stFlrSF','GarageCars','Street','Alley','LotConfig','BldgType','HouseStyle']
X = all_data[all_data.GarageQual.notnull()][init_f]
y = all_data[all_data.GarageQual.notnull()]['GarageQual']
X_dumm = pd.get_dummies(X)
selector = SelectKBest(chi2, k=12)
a = selector.fit(X_dumm, y)

sorted_index = np.argsort(a.scores_)[-10:]

# plt.style.use({'figure.figsize':(30, 14)})
# sns.barplot(x=X_dumm.columns[sorted_index], y=np.sort(np.array(a.scores_))[:], ci=0)
# plt.xticks(rotation=90)
# plt.show()

'''对 GarageQual 影响比较大的列'''
selected_columns = X_dumm.columns[sorted_index]

X = all_data[all_data.GarageQual.notnull()]
X = pd.get_dummies(X)
X = X[selected_columns]
y = all_data[all_data.GarageQual.notnull()]['GarageQual']

X_pred = all_data[all_data.GarageQual.isnull()]
X_pred = pd.get_dummies(X_pred)
X_pred = X_pred[selected_columns]

'''搜索最优参数'''
# pipe = Pipeline([
#     ('rfc', RandomForestClassifier(random_state=1))
# ])
# param_grid = {
#     'rfc__n_estimators': list(range(3, 60, 2)),
#     'rfc__max_depth': list(range(2, 20, 2))
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, n_jobs=-1, scoring='accuracy')
# gsearch.fit(X, y)
# ic(gsearch.best_params_, gsearch.best_score_)

'''评估模型'''
pipe = Pipeline([
    ('rfc', RandomForestClassifier(random_state=1, max_depth=6, n_estimators=41))
])
# scores = cross_val_score(estimator=pipe, X=X, y=y, cv=10, n_jobs=-1, scoring='accuracy')
# ic(scores.mean(), np.std(scores))  # 0.944927536231884  0.00354998513446839
pipe.fit(X, y)
y_pred = pipe.predict(X_pred)
all_data.loc[all_data.GarageQual.isnull(), 'GarageQual'] = y_pred

# 通过随机森林填补 GarageCond 缺失值
X = all_data[all_data.GarageCond.notnull()]
X = pd.get_dummies(X)
X = X[selected_columns]
y = all_data[all_data.GarageCond.notnull()]['GarageCond']

X_pred = all_data[all_data.GarageCond.isnull()]
X_pred = pd.get_dummies(X_pred)
X_pred = X_pred[selected_columns]

'''评估模型'''
pipe = Pipeline([
    ('rfc', RandomForestClassifier(random_state=1, max_depth=6, n_estimators=41))
])
# scores = cross_val_score(estimator=pipe, X=X, y=y, cv=10, n_jobs=-1, scoring='accuracy')
# ic(scores.mean(), np.std(scores))  # 0.9612318840579711  0.0023199725497944725
pipe.fit(X, y)
y_pred = pipe.predict(X_pred)
all_data.loc[all_data.GarageCond.isnull(), 'GarageCond'] = y_pred

# 填补 GarageFinish 缺失值
g = all_data.groupby('GarageCond')['GarageFinish'].value_counts().reset_index(name='count')

def fill_gf(row):
    a = g[g.GarageCond == row.GarageCond]['GarageFinish']
    return a.values[0]

all_data.loc[all_data.GarageFinish.isnull(), 'GarageFinish'] = all_data[all_data.GarageFinish.isnull()].apply(fill_gf, axis=1)

# 填补 GarageYrBlt 缺失值
'''将异常值设为nan'''
all_data.loc[all_data.GarageYrBlt >= 2020, 'GarageYrBlt'] = np.nan

'''第一次填补'''
my_df = all_data.groupby('YearBuilt')['GarageYrBlt'].value_counts().reset_index(name='count')

def fill_gyb(row):
    w = my_df[my_df.YearBuilt == row.YearBuilt]['GarageYrBlt']
    if len(w.values) == 0:
        return np.nan
    return w.values[0]

all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageYrBlt'] = all_data[all_data.GarageYrBlt.isnull()].apply(fill_gyb, axis=1)

'''第二次填补'''
my_df2 = all_data.groupby('YearRemodAdd')['GarageYrBlt'].value_counts().reset_index(name='count')

def fill_gyb2(row):
    w2 = my_df2[my_df2.YearRemodAdd == row.YearRemodAdd]['GarageYrBlt']
    return w2.values[0]

all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageYrBlt'] = all_data[all_data.GarageYrBlt.isnull()].apply(fill_gyb2, axis=1)

# 填补 GarageType 缺失值
init_f = ['MSSubClass','MSZoning','OverallCond','YearBuilt','YearRemodAdd','Foundation','BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','1stFlrSF','GarageCars','Street','Alley','LotConfig','BldgType','HouseStyle']

X = all_data[all_data.GarageType.notnull()][init_f]
y = all_data[all_data.GarageType.notnull()]['GarageType']

X_dumm = pd.get_dummies(X)
selector = SelectKBest(chi2)

a = selector.fit(X_dumm, y)
sorted_index = np.argsort(a.scores_)[-13:]

'''对 GarageType 影响比较大的列'''
selected_columns = X_dumm.columns[sorted_index]

X_gt = all_data[all_data.GarageType.notnull()]
X_gt = pd.get_dummies(X_gt)
X_gt = X_gt[selected_columns]
y_gt = all_data[all_data.GarageType.notnull()]['GarageType']

X_gt_test = all_data[all_data.GarageType.isnull()]
X_gt_test = pd.get_dummies(X_gt_test)
X_gt_test = X_gt_test[selected_columns]


'''搜索最优参数'''
# pipe = Pipeline([
#     ('rfc', RandomForestClassifier(random_state=3))
# ])
# param_grid = {
#     'rfc__n_estimators': list(range(3, 60, 2)),
#     'rfc__max_depth': list(range(2, 20, 2)),
#     'rfc__max_features': ['auto', 'sqrt', 'log2']
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, n_jobs=-1, scoring='accuracy', verbose=2)
# gsearch.fit(X_gt, y_gt)
# ic(gsearch.best_params_, gsearch.best_score_)

'''评估模型'''
pipe_best = Pipeline([
    ('rfc', RandomForestClassifier(random_state=3, max_depth=10, n_estimators=33))
])
# scores = cross_val_score(estimator=pipe_best, X=X_gt, y=y_gt, cv=10, n_jobs=-1, scoring='accuracy')
# ic(np.mean(scores), np.std(scores))

pipe_best.fit(X_gt, y_gt)
y_gt_pred = pipe_best.predict(X_gt_test)
all_data.loc[all_data.GarageType.isnull(), 'GarageType'] = y_gt_pred

# 填补 SaleType 缺失值
g = all_data.groupby('SaleCondition')['SaleType'].value_counts().reset_index(name='count')

def fill_st(row):
    a = g[g.SaleCondition == row.SaleCondition]['SaleType']
    return a.values[0]

all_data.loc[all_data.SaleType.isnull(), 'SaleType'] = all_data[all_data.SaleType.isnull()].apply(fill_st, axis=1)

# 填补 FireplaceQu 缺失值
'''第一次'''
g = all_data.groupby('Heating')['FireplaceQu'].value_counts().reset_index(name='count')

def fill_fq(row):
    a = g[g.Heating == row.Heating]['FireplaceQu']
    return a.values[0]

all_data.loc[(all_data.FireplaceQu.isnull()) & (all_data.Heating.isin(['GasA', 'OthW', 'Grav'])), 'FireplaceQu'] = all_data[(all_data.FireplaceQu.isnull()) & (all_data.Heating.isin(['GasA', 'OthW', 'Grav']))].apply(fill_fq, axis=1)

'''第二次'''
g2 = all_data.groupby('HeatingQC')['FireplaceQu'].value_counts().reset_index(name='count')

def fill_fq2(row):
    a = g2[g2.HeatingQC == row.HeatingQC]['FireplaceQu']
    return a.values[0]

all_data.loc[(all_data.FireplaceQu.isnull()), 'FireplaceQu']=all_data[(all_data.FireplaceQu.isnull())].apply(fill_fq2, axis=1)

# 增加新特征 HasPool
def add_HasPool(row):
    if row.PoolArea == 0:
        return 0
    else:
        return 1
all_data['HasPool'] = all_data.apply(add_HasPool, axis=1)

# 填补Fence
'''第一次填补'''
g = all_data.groupby(by=['HasPool','OverallCond','Street'])['Fence'].value_counts().reset_index(name='count')
def fill_Fence(row):
    a = g[(g.HasPool==row.HasPool) & (g.OverallCond==row.OverallCond) & (g.Street==row.Street)]['Fence']
    if len(a.values) == 0:
        return np.nan
    else:
        return a.values[0]
all_data.loc[all_data.Fence.isnull(), 'Fence']=all_data[all_data.Fence.isnull()].apply(fill_Fence, axis=1)

'''第二次填补'''
g = all_data.groupby(by=['HasPool','OverallCond','MSZoning'])['Fence'].value_counts().reset_index(name='count')
def fill_Fence2(row):
    a = g[(g.HasPool==row.HasPool) & (g.OverallCond==row.OverallCond) & (g.MSZoning==row.MSZoning)]['Fence']
    return a.values[0]
all_data.loc[all_data.Fence.isnull(), 'Fence'] = all_data[all_data.Fence.isnull()].apply(fill_Fence2, axis=1)

# 处理 WoodDeckSF 异常值
all_data.loc[all_data.WoodDeckSF == 1424, 'WoodDeckSF'] = 100

# # 将 PoolArea 变成离散型变量
# def transform_pa(row):
#     if row.PoolArea >= 0 and row.PoolArea <= 480:
#         return 0
#     elif row.PoolArea > 578 and row.PoolArea <= 650:
#         return 0
#     else:
#         return 1
# all_data['PoolArea2'] = all_data.apply(transform_pa, axis=1)


# 处理ExterQual，ExterCond
d = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1
}
all_data['ExterQual2'] = all_data['ExterQual'].map(d)
all_data['ExterCond2'] = all_data['ExterCond'].map(d)
all_data['ExterAssess'] = all_data['ExterQual2'] + all_data['ExterCond2']

# 处理BsmtQual，BsmtCond
d = {
    'Ex': 6,
    'Gd': 5,
    'TA': 4,
    'Fa': 3,
    'Po': 2,
    'NA': 1,
}
all_data['BsmtQual2'] = all_data['BsmtQual'].map(d)
all_data['BsmtCond2'] = all_data['BsmtCond'].map(d)
all_data['BsmtAssess'] = all_data['BsmtQual2'] + all_data['BsmtCond2']

# 处理GarageQual，GarageCond
d = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1
}
all_data['GarageQual2'] = all_data['GarageQual'].map(d)
all_data['GarageCond2'] = all_data['GarageCond'].map(d)
all_data['GarageAssess'] = all_data['GarageQual2'] + all_data['GarageCond2']

# 处理BsmtFullBath，BsmtHalfBath *
all_data['FullHalfBath'] = all_data['FullBath'] + all_data['HalfBath']

# 处理BsmtFinType1，BsmtFinType2
# d = {
#     'GLQ': 6,
#     'ALQ': 5,
#     'BLQ': 4,
#     'Rec': 3,
#     'LwQ': 2,
#     'Unf': 1,
# }
# all_data['BsmtFinType1_New'] = all_data['BsmtFinType1'].map(d)
# all_data['BsmtFinType2_New'] = all_data['BsmtFinType2'].map(d)
# all_data['BsmtFinType_Total'] = all_data['BsmtFinType1_New'] + all_data['BsmtFinType2_New']
all_data['BsmtFinSF_Total'] = all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalPorch'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']
all_data['OverallAssess'] = all_data['OverallQual'] + all_data['OverallCond']

################################################### 特征工程完毕


'''
去掉：OverallCond,MasVnrType,MasVnrArea,BsmtFullBath,KitchenAbvGr,PoolArea,SaleType,Exterior1st,BsmtExposure,MiscVal
'''
# high_features = ['SalePrice_Log','MSSubClass','MSZoning','LotFrontage','LotArea','Street','LandContour','Utilities','Neighborhood','Condition1','BldgType','HouseStyle','OverallQual','YearBuilt','ExterQual','Foundation','BsmtCond','BsmtFinType1','BsmtFinSF1','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','LowQualFinSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageCars','GarageQual','PavedDrive','WoodDeckSF','PoolArea2','SaleCondition']
# add_features = ['LotConfig','RoofMatl','2ndFlrSF','OpenPorchSF','Fence']


'''相关性高的特征'''
# 数值型
numerical_features = ['LotFrontage','LotArea','YearBuilt','MasVnrArea','TotalSF','TotalBsmtSF','LowQualFinSF','GrLivArea','WoodDeckSF','PoolArea','TotalPorch','MiscVal']
# 数值（实际是类别型）
shuzhi_features = ['OverallAssess','TotRmsAbvGrd','Fireplaces','FullHalfBath','MSSubClass','KitchenAbvGr','ExterAssess','BsmtAssess','GarageAssess']
# 类别型
non_numerical_features = ['Foundation','BsmtExposure','Condition1','Functional','BsmtFinSF_Total','BsmtUnfSF','SaleCondition','CentralAir','MasVnrType','BsmtFullBath','Exterior1st','BldgType','MSZoning','FireplaceQu','Neighborhood','Heating','PavedDrive','Fence','GarageType','LotConfig','HouseStyle','RoofMatl','SaleType','Street','SalePrice_Log','Electrical','GarageCars','KitchenQual','LandContour','HeatingQC','BsmtFinType1']

# # 转换数据
# all_data['OverallAssess'] = all_data['OverallAssess'].astype(str)
# all_data['TotRmsAbvGrd'] = all_data['TotRmsAbvGrd'].astype(str)
# all_data['Fireplaces'] = all_data['Fireplaces'].astype(str)
# all_data['FullBath'] = all_data['FullBath'].astype(str)
# all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
# all_data['KitchenAbvGr'] = all_data['KitchenAbvGr'].astype(str)


'''对数值型变量降低偏度'''
from scipy.stats import skew
from scipy.special import boxcox1p
skewed_features = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)  # series
# 偏度较大的特征
skewed_features = skewed_features[abs(skewed_features) > 0.75]
skewed_features_name = skewed_features.index

lam = 0.15 # 超参数
for feat in skewed_features_name:
    tranformer_feat = boxcox1p(all_data[feat], lam)
    all_data[feat] = tranformer_feat

# 选取的特征
new_features = numerical_features + non_numerical_features + shuzhi_features

# miss_features = ['Alley','PoolQC','MiscFeature']

all_data2 = all_data[new_features]

# 训练，测试数据
train_data = all_data2[all_data2.SalePrice_Log.notnull()]
test_data = all_data2[all_data2.SalePrice_Log.isnull()]

# 训练，评估，预测
X_train = train_data.drop('SalePrice_Log', axis=1)
X_train_dumm = pd.get_dummies(X_train)
y = train_data['SalePrice_Log']

'''预测数据'''
X_test = test_data.drop('SalePrice_Log', axis=1)
X_test_dumm = pd.get_dummies(X_test)

'''处理训练数据和测试数据维度不相同问题'''
train_lst = set(list(X_train_dumm.columns))
test_lst = set(list(X_test_dumm.columns))
sub_columns_train = list(train_lst - test_lst)
sub_columns_test = list(test_lst - train_lst)
X_train_dumm.drop(sub_columns_train, axis=1, inplace=True)
X_test_dumm.drop(sub_columns_test, axis=1, inplace=True)

# total_dumm = pd.concat([X_train_dumm, X_test_dumm], axis=0)

"""
ic| X_train_dumm.shape: (1458, 190)
ic| X_test_dumm.shape: (1459, 190)
"""

'''归一化'''
# std_scaler = RobustScaler()
# std_scaler.fit(total_dumm)
# X_train_dumm = std_scaler.transform(X_train_dumm)
# X_test_dumm = std_scaler.transform(X_test_dumm)


# total_dumm = std_scaler.transform(total_dumm)

'''PCA降维'''
# pca = PCA(n_components=25)
# pca.fit(total_dumm)
# X_train_dumm = pca.transform(X_train_dumm)
# X_test_dumm = pca.transform(X_test_dumm)


# Lasso 回归

# pipe = Pipeline([
#     ('std_scaler', RobustScaler()),
#     ('lasso', Lasso(random_state=1))
# ])
# param_grid = {
#     'lasso__alpha': [0.0001,0.0003,0.0005,0.0007,0.0009],
#     'lasso__tol': [1e-4,1e-3,1e-2],
#     'lasso__selection': ['cyclic','random'],
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm, y)
# ic(gsearch.best_params_, gsearch.best_score_)

pipe_best_lasso = Pipeline([
    ('std_scaler', RobustScaler()),
    ('lasso', Lasso(random_state=1, alpha=0.0006, selection='random', tol=0.0001))
])

'''交叉验证'''
# scores = cross_val_score(pipe_best_lasso, X_train_dumm, y, scoring="neg_mean_squared_error", cv=5)
# ic(np.mean(scores), np.std(scores))


'''岭回归'''
# pipe = Pipeline([
#     ('std_scaler', RobustScaler()),
#     ('ridge', Ridge(random_state=1))
# ])
# param_grid = {
#     'ridge__alpha': list(range(2,20)),
#     'ridge__tol': [1e-3],
#     # 'ridge__solver':['auto','svd','cholesky','sparse_cg','lsqr','sag']
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm, y)
# ic(gsearch.best_params_, gsearch.best_score_)


'''弹性网络回归'''
# pipe = Pipeline([
#     ('std_scaler', RobustScaler()),
#     ('en', ElasticNet(random_state=1))
# ])
# param_grid = {
#     'en__alpha': [0.0004],
#     'en__l1_ratio': [0.9],
#     'en__tol': [1e-3,1e-4,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5],
#     'en__selection': ['cyclic','random']
#     # 'ridge__solver':['auto','svd','cholesky','sparse_cg','lsqr','sag']
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm, y)
# ic(gsearch.best_params_, gsearch.best_score_)

pipe_best_en = Pipeline([
    ('std_scaler', RobustScaler()),
    ('en', ElasticNet(random_state=1, alpha=0.00065, l1_ratio=0.9, selection='cyclic', tol=5e-05))
])
# scores = cross_val_score(pipe_best_en, X_train_dumm, y, scoring="neg_mean_squared_error", cv=5)
# ic(np.mean(scores), np.std(scores))


# GradientBoostingRegressor 算法  kaggle得分：0.15578  0.15412

'''搜索最佳参数'''
# pipe = Pipeline([
#     ('std_scaler', RobustScaler()),
#     ('gbr', GradientBoostingRegressor(random_state=2))
# ])
# param_grid = {
#     'gbr__loss': ['ls'],
#     'gbr__n_estimators': [40,80,120],
#     'gbr__learning_rate': [0.05],
#     'gbr__max_depth': list(range(2,20,2)),
#     'gbr__max_features': [0.3,0.1,0.5,0.7,0.2],
#     'gbr__min_samples_leaf': [1,2,3,4],
#     # 'gbr__alpha': [0.95],
# }
# # param_grid = {
# #     # 'gbr__loss': ['ls', 'lad', 'huber'],
# #     'gbr__n_estimators': [100,200,300],
# #     'gbr__learning_rate': [0.01, 0.02, 0.03],
# #     'gbr__max_depth': list(range(4, 16, 2)),
# #     'gbr__max_features': [0.3, 0.5, 0.7],
# #     'gbr__min_samples_leaf': [40, 45, 50]
# # }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm, y)
# ic(gsearch.best_params_, gsearch.best_score_)

pipe_best_gbr = Pipeline([
    ('std_scaler', RobustScaler()),
    ('gbr', GradientBoostingRegressor(random_state=2, n_estimators=600, learning_rate=0.05, max_depth=4, max_features=0.2, min_samples_leaf=3, loss='ls', alpha=0.95, ))
])
'''交叉验证'''
# scores = cross_val_score(pipe_best_gbr, X_train_dumm, y, scoring="neg_mean_squared_error", cv=5)
# ic(np.mean(scores), np.std(scores))

# SVR 算法  kaggle得分:0.14011  0.14097

# pipe = Pipeline([
#     ('std_scaler', RobustScaler()),
#     ('svr', SVR())
# ])
# param_grid = {
#     'svr__C': [60],
#     # 'svr__tol': [3,2,0.1,0.15,1e-2]
#     # 'svr__degree': [1, 2, 3, 4, 5, 6],
#     'svr__gamma': [0.0006],
#     'svr__kernel': ['rbf'],  # linear 特别花费时间
#     'svr__tol': [0.01],
#     'svr__epsilon': [0.01,0.012,0.014,0.016,0.018,0.02,0.022,]
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm, y)
# ic(gsearch.best_params_, gsearch.best_score_)

pipe_best_svr = Pipeline([
    ('std_scaler', RobustScaler()),
    ('svr', SVR(C=57, gamma=0.0003, kernel='rbf', epsilon=0.018, tol=0.01))
])
# scores = cross_val_score(estimator=pipe_best_svr, X=X_train_dumm, y=y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# ic(np.mean(scores), np.std(scores))

# '''Stack Model'''
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#
#     def __init__(self, models):
#         self.models = models
#
#     # we define clones of the original models to fit the data in
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
#         # Train cloned base models
#         for model in self.models_:
#             model.fit(X, y)
#
#         return self
#
#     # Now we do the predictions for cloned models and average them
#     def predict(self, X):
#         predictions = np.column_stack([
#             model.predict(X) for model in self.models_
#         ])
#         return np.mean(predictions, axis=1)
#
# averaged_models = AveragingModels(models=(pipe_best_en, pipe_best_svr, pipe_best_lasso))
# scores = cross_val_score(estimator=averaged_models, X=X_train_dumm, y=y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# ic(np.mean(scores), np.std(scores))


# 集成学习
voting_reg = VotingRegressor(
    estimators=[
        ('svr', pipe_best_svr),
        ('en', pipe_best_en),
        ('lasso', pipe_best_lasso),
    ],
    weights=[0.5,0.3,0.2]
)

'''交叉验证'''
# scores = cross_val_score(estimator=voting_reg, X=X_train_dumm, y=y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# ic(np.mean(scores), np.std(scores))

'''训练'''
voting_reg.fit(X_train_dumm, y)
y_pred = voting_reg.predict(X_test_dumm)
y_pred = np.exp(y_pred)
'''生成csv文件'''
submission = pd.DataFrame({'Id': SubmitId, 'SalePrice': np.round(y_pred, 1)})
submission.to_csv('./submit/sub13.csv', index=False)