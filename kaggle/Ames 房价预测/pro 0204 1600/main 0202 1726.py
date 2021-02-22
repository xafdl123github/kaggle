import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

SubmitId = test_data['Id']

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

# 将 PoolArea 变成离散型变量
def transform_pa(row):
    if row.PoolArea >= 0 and row.PoolArea <= 480:
        return 0
    elif row.PoolArea > 578 and row.PoolArea <= 650:
        return 0
    else:
        return 1
all_data['PoolArea2'] = all_data.apply(transform_pa, axis=1)

################################################### 特征工程完毕

# all_data.drop(['Alley','PoolQC','Fence','MiscFeature','Id'], inplace=True, axis=1)
# 去掉：YearRemodAdd，BedroomAbvGr

'''
去掉：OverallCond,MasVnrType,MasVnrArea,BsmtFullBath,KitchenAbvGr,PoolArea,SaleType,Exterior1st,BsmtExposure,MiscVal
'''

'''相关性高的特征'''
# high_features = ['SalePrice','MSSubClass','MSZoning','LotFrontage','LotArea','Street','LandContour','Utilities','Neighborhood','Condition1','BldgType','HouseStyle','OverallQual','YearBuilt','ExterQual','Foundation','BsmtCond','BsmtFinType1','BsmtFinSF1','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','LowQualFinSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageCars','GarageQual','PavedDrive','WoodDeckSF','PoolArea2','SaleCondition']
# add_features = ['LotConfig','RoofMatl','2ndFlrSF','OpenPorchSF','Fence']

'''相关性高的特征'''
high_features = ['SalePrice','MSSubClass','MSZoning','LotFrontage','LotArea','Street','LandContour','Utilities','Neighborhood','Condition1','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','MasVnrType','MasVnrArea','ExterQual','Foundation','BsmtCond','BsmtFinType1','BsmtFinSF1','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','FullBath','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageCars','GarageQual','PavedDrive','WoodDeckSF','PoolArea','SaleType','SaleCondition']
add_features = ['LotConfig','RoofMatl','Exterior1st','BsmtExposure','2ndFlrSF','OpenPorchSF','Fence','MiscVal']

# select_features_by_corr = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']

new_features = high_features + add_features

# miss_features = ['Alley','PoolQC','MiscFeature']

all_data2 = all_data[new_features]

# 训练，测试数据
train_data = all_data2[all_data2.SalePrice.notnull()]
test_data = all_data2[all_data2.SalePrice.isnull()]

# 训练，评估，预测
X_train = train_data.drop('SalePrice', axis=1)
X_train_dumm = pd.get_dummies(X_train)
y = train_data['SalePrice']

# X_train = X_dumm

'''预测数据'''
X_test = test_data.drop('SalePrice', axis=1)
X_test_dumm = pd.get_dummies(X_test)

'''处理训练数据和测试数据维度不相同问题'''
train_lst = set(list(X_train_dumm.columns))
test_lst = set(list(X_test_dumm.columns))
sub_columns = list(train_lst - test_lst)
X_train_dumm.drop(sub_columns, axis=1, inplace=True)

'''PCA降维'''
# pca = PCA(n_components=20)
# pca.fit(X_train_dumm)
# X_train_dumm_reduction = pca.transform(X_train_dumm)
# X_test_dumm_reduction = pca.transform(X_test_dumm)

# 随机森林 kaggle得分：0.16203 0.16140

'''搜索最佳参数'''
# pipe = Pipeline([
#     ('std_scaler', StandardScaler()),
#     ('rfr', RandomForestRegressor(random_state=1))
# ])
# param_grid = {
#     'rfr__n_estimators': list(range(10, 60, 2)),
#     'rfr__max_depth': list(range(10, 30, 2)),
#     'rfr__max_features': [0.3,0.5,0.7]
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm_reduction, y)
# ic(gsearch.best_params_, gsearch.best_score_)


'''最优模型'''
pipe_best_forest = Pipeline([
    ('std_scaler', StandardScaler()),
    ('rfr', RandomForestRegressor(random_state=1, n_estimators=40, max_depth=16, max_features='auto'))
])
# scores = cross_val_score(estimator=pipe_best_forest, X=X_train_reduction, y=y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# ic(np.mean(scores), np.std(scores))


# GradientBoostingRegressor 算法  kaggle得分：0.15578  0.15412

'''搜索最佳参数'''
# pipe = Pipeline([
#     ('pca', PCA()),
#     ('std_scaler', StandardScaler()),
#     ('gbr', GradientBoostingRegressor(random_state=2))
# ])
# param_grid = {
#     'pca__n_components': [16,18,20],
#     'gbr__loss': ['huber'],
#     'gbr__n_estimators': [850],
#     'gbr__learning_rate': [0.03],
#     'gbr__max_depth': list(range(4, 16, 2)),
#     'gbr__max_features': [0.5],
#     'gbr__min_samples_leaf': [45],
#     'gbr__alpha': [0.95],
# }
# # param_grid = {
# #     'gbr__loss': ['ls', 'lad', 'huber'],
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
    ('pca', PCA(n_components=20)),
    ('std_scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor(random_state=2, n_estimators=850, learning_rate=0.03, max_depth=6, max_features=0.5, min_samples_leaf=45, loss='huber', alpha=0.95, ))
])


# SVR 算法  kaggle得分:0.14011  0.14097

# pipe = Pipeline([
#     ('std_scaler', StandardScaler()),
#     ('svr', SVR())
# ])
# param_grid = {
#     # 'svr__C': [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,2,5,7,12,26],
#     # 'svr__tol': [3,2,1,0.1,0.15,1e-2]
#     # 'svr__degree': [1, 2, 3, 4, 5, 6],
#     'svr__gamma': [0.01],
#     'svr__C': [1e6],
#     'svr__kernel': ['rbf'],  # linear 特别花费时间
#     'svr__tol': [1e4],
#     'svr__epsilon': [0.15]
# }
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
# gsearch.fit(X_train_dumm_reduction, y)
# ic(gsearch.best_params_, gsearch.best_score_)

pipe_best_svr = Pipeline([
    ('std_scaler', StandardScaler()),
    ('svr', SVR(C=1e6, gamma=0.01, kernel='rbf', epsilon=0.15, tol=1e4))
])

# 集成学习
voting_reg = VotingRegressor(
    estimators=[
        # ('forest', pipe_best_forest),
        ('gbr', pipe_best_gbr),
        ('svr', pipe_best_svr),
    ]
)

'''交叉验证'''
# scores = cross_val_score(estimator=voting_reg, X=X_train_dumm_reduction, y=y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# ic(np.mean(scores), np.std(scores))

'''训练'''
pipe_best_gbr.fit(X_train_dumm, y)
y_pred = pipe_best_gbr.predict(X_test_dumm)
'''生成csv文件'''
submission = pd.DataFrame({'Id': SubmitId, 'SalePrice': np.round(y_pred, 1)})
submission.to_csv('./submit/sub24.csv', index=False)