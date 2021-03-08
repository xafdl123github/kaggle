import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic
from sklearn.preprocessing import LabelEncoder
import time
from itertools import product
from icecream import ic


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sales_train = pd.read_csv('./data/sales_train.csv')
test = pd.read_csv('./data/test.csv')   # (214200, 3)

# 计算每个商品每个月的销售量，假如某个商品在某个月没有数据，则填充0（即这个月的销售量为0）
sales_by_item_id = sales_train.pivot_table(index=['item_id'], values=['item_cnt_day'], columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)   # 去掉第一层索引
sales_by_item_id.columns.values[0] = 'item_id'
sales_by_item_id = sales_by_item_id.rename_axis(None, axis=1)

# 获取最近6个月销售量为0的数据
six_zero = sales_by_item_id[(sales_by_item_id['28'] == 0) & (sales_by_item_id['29'] == 0) & (sales_by_item_id['30'] == 0) & (sales_by_item_id['31'] == 0) & (sales_by_item_id['32'] == 0) & (sales_by_item_id['33'] == 0)]
six_zero_item_id = list(six_zero['item_id'].values)   # item_id列表
# test.loc[test.item_id.isin(six_zero_item_id), 'item_cnt_month'] = 0  # 将test数据中（最近六个月销量为0）的数据月销量设为0，有7812个

# 计算每个商店每个月的销量
sales_by_shop_id = sales_train.pivot_table(index=['shop_id'], values=['item_cnt_day'], aggfunc=np.sum, fill_value=0, columns='date_block_num').reset_index()
sales_by_shop_id.columns = sales_by_shop_id.columns.droplevel().map(str)    # 将两层column转化为一层column,保留下层column
sales_by_shop_id.columns.values[0] = 'shop_id'
sales_by_shop_id = sales_by_shop_id.rename_axis(None, axis=1)   # 将列方向的轴重命名为none

# zero = sales_train[sales_train.date_block_num==0]
# ic(zero.shop_id.unique(), len(zero.item_id.unique()), len(zero.shop_id.unique()), len(zero.shop_id.unique()) * len(zero.item_id.unique()))
# ic(sales_train.shop_id.unique(), len(sales_train.item_id.unique()), len(sales_train.shop_id.unique()), len(sales_train.shop_id.unique()) * len(sales_train.item_id.unique()))

"""组合date_block_num,shop_id,item_id(部分) 总量：10913850"""
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = sales_train[sales_train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix.sort_values(cols, inplace=True)  # 排序
sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']    # 某一天的销售额

# 分组
groupby = sales_train.groupby(['shop_id','item_id','date_block_num']).agg({'item_cnt_day': 'sum'}).reset_index()
groupby = groupby.rename(columns={'item_cnt_day': 'item_cnt_month'})
matrix = matrix.merge(groupby, on=['date_block_num','shop_id','item_id'], how='left')
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0, 20)
matrix['item_cnt_month'] = matrix['item_cnt_month'].astype(np.float16)

# test数据
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

# 合并matrix,test
matrix = pd.concat([matrix, test[cols]], ignore_index=True, axis=0)
matrix['item_cnt_month'].fillna(0, inplace=True)

# 商品信息
items = pd.read_csv('./data/items.csv')
items = items[['item_id', 'item_category_id']]
matrix = pd.merge(left=matrix, right=items, on='item_id', how='left')  # 合并

# 商品类别
le = LabelEncoder()
categories = pd.read_csv('./data/item_categories.csv')
categories['split'] = categories['item_category_name'].str.split('-')
categories['type'] = categories['split'].map(lambda x:x[0].strip())
categories['subtype'] = categories['split'].map(lambda x:x[1].strip() if len(x)>1 else x[0].strip())
categories = categories[['item_category_id','type','subtype']]
categories['cat_type_code'] = le.fit_transform(categories['type'])
categories['cat_subtype_code'] = le.fit_transform(categories['subtype'])
matrix = pd.merge(left=matrix, right=categories[['item_category_id','cat_type_code','cat_subtype_code']], on='item_category_id', how='left')    # 合并

# 商店信息
shops = pd.read_csv('./data/shops.csv')
shops['split']=shops.shop_name.str.split(' ')
shops['shop_city'] = shops['split'].map(lambda x:x[0])
shops['shop_city_code'] = le.fit_transform(shops['shop_city'])

def st(name):
    if 'ТЦ' in name or 'ТРЦ' in name:
        shopt = 'ТЦ'
    elif 'ТК' in name:
        shopt = 'ТК'
    elif 'ТРК' in name:
        shopt = 'ТРК'
    elif 'МТРЦ' in name:
        shopt = 'МТРЦ'
    else:
        shopt = 'UNKNOWN'
    return shopt
shops['shop_type'] = shops['shop_name'].apply(st)

shops.loc[shops.shop_id == 21, 'shop_type'] = 'МТРЦ'   # 修正
shops['shop_type_code'] = le.fit_transform(shops['shop_type'])
matrix = pd.merge(left=matrix, right=shops[['shop_id','shop_city_code','shop_type_code']], on='shop_id', how='left')    # 合并
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['cat_type_code'] = matrix['cat_type_code'].astype(np.int8)
matrix['cat_subtype_code'] = matrix['cat_subtype_code'].astype(np.int8)
matrix['shop_city_code'] = matrix['shop_city_code'].astype(np.int8)
matrix['shop_type_code'] = matrix['shop_type_code'].astype(np.int8)


"""历史信息"""

def lag_features(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id',col+'_lag_'+str(i)]
        shifted['date_block_num'] = shifted['date_block_num'] + i
        df = pd.merge(left=df, right=shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

matrix = lag_features(matrix, [1,2,3,6,12], 'item_cnt_month')

# 月销量（所有商品）
group = matrix.groupby('date_block_num').agg({'item_cnt_month': 'mean'}).reset_index()
group.columns = ['date_block_num', 'date_avg_item_cnt']
matrix = pd.merge(left=matrix, right=group, on='date_block_num', how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_avg_item_cnt')
matrix.drop('date_avg_item_cnt', axis=1, inplace=True)

# 月销量（每一件商品）
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)
matrix = pd.merge(left=matrix, right=group, on=['date_block_num','item_id'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop('date_item_avg_item_cnt', axis=1, inplace=True)

# 月销量（每个商店 ）
group = matrix.groupby(['date_block_num','shop_id']).agg({'item_cnt_month': 'mean'})
group.columns = ['date_shop_avg_item_cnt']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num','shop_id'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop('date_shop_avg_item_cnt', axis=1, inplace=True)



# 月销量（每个类别）
group = matrix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month': 'mean'})
group.columns = ['date_cat_avg_item_cnt']
group = group.reset_index()
matrix=pd.merge(left=matrix, right=group, on=['date_block_num','item_category_id'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_cat_avg_item_cnt')
matrix.drop('date_cat_avg_item_cnt', axis=1, inplace=True)

# 月销量（商品类别-商店）
group = matrix.groupby(['date_block_num','item_category_id','shop_id']).agg({'item_cnt_month': 'mean'})
group.columns = ['date_cat_shop_avg_item_cnt']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num','item_category_id','shop_id'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_cat_shop_avg_item_cnt')
matrix.drop('date_cat_shop_avg_item_cnt', axis=1, inplace=True)

# 月销量（商品大类）
group = matrix.groupby(['date_block_num','cat_type_code']).agg({'item_cnt_month': 'mean'})
group.columns = ['date_type_avg_item_cnt']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num','cat_type_code'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_type_avg_item_cnt')
matrix.drop('date_type_avg_item_cnt', axis=1, inplace=True)

# 月销量（商品-商品大类） ++++++++++++ 和 月销量（商品）是重复的，因为每一个商品，类别是确定的，大类也是确定的
group = matrix.groupby(['date_block_num', 'item_id', 'cat_type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_type_avg_item_cnt']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num', 'item_id', 'cat_type_code'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_item_type_avg_item_cnt')
matrix.drop('date_item_type_avg_item_cnt', axis=1, inplace=True)

# 月销量（商店城市）
group = matrix.groupby(['date_block_num','shop_city_code']).agg({'item_cnt_month': 'mean'})
group.columns = ['date_city_avg_item_cnt']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num','shop_city_code'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_city_avg_item_cnt')
matrix.drop('date_city_avg_item_cnt', axis=1, inplace=True)

# 月销量（商品-商店城市）
group = matrix.groupby(['date_block_num', 'item_id', 'shop_city_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_city_avg_item_cnt']
group = group.reset_index()
matrix=pd.merge(left=matrix, right=group, on=['date_block_num', 'item_id', 'shop_city_code'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_item_city_avg_item_cnt')
matrix.drop('date_item_city_avg_item_cnt', axis=1, inplace=True)



# 趋势特征
group = sales_train.groupby('item_id').agg({'item_price': 'mean'})
group.columns = ['item_avg_item_price']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on='item_id', how='left')

group = sales_train.groupby(['date_block_num','item_id']).agg({'item_price': 'mean'})
group.columns = ['date_item_avg_item_price']
group = group.reset_index()
matrix=pd.merge(left=matrix, right=group, on=['date_block_num','item_id'], how='left')

matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

# 计算matrix中商品的历史价格
lags = [1,2,3,4,5,6,12]
matrix = lag_features(matrix, lags, 'date_item_avg_item_price')
for i in lags:
    matrix['delta_price_lag_'+str(i)] = (matrix['date_item_avg_item_price_lag_' + str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if pd.notnull(row['delta_price_lag_'+str(i)]):  # 如果不是NaN
            return row['delta_price_lag_'+str(i)]
    return 0   #  如果delta_price_lag_都为空，那么将趋势设为0，0代表没有趋势

matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)

features_to_drop = ['item_avg_item_price','date_item_avg_item_price']
for i in lags:
    features_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    features_to_drop += ['delta_price_lag_'+str(i)]
matrix.drop(features_to_drop, axis=1, inplace=True)

# 每个月的天数
matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days)
matrix['days'] = matrix['days'].astype(np.int8)

# 开始销量
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

# 月销量（商店类型）
group = matrix.groupby(['date_block_num','shop_type_code']).agg({'item_cnt_month': 'mean'})
group.columns = ['date_shoptype_avg_item_cnt']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num','shop_type_code'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_shoptype_avg_item_cnt')
matrix.drop('date_shoptype_avg_item_cnt', axis=1, inplace=True)

# 月销量（商品-商店类型）
group = matrix.groupby(['date_block_num', 'item_id', 'shop_type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_shoptype_avg_item_cnt']
group = group.reset_index()
matrix=pd.merge(left=matrix, right=group, on=['date_block_num', 'item_id', 'shop_type_code'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_item_shoptype_avg_item_cnt')
matrix.drop('date_item_shoptype_avg_item_cnt', axis=1, inplace=True)

# 月销量（商店-商品）
group = matrix.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shopitem_avg_item_cnt' ]
group.reset_index(inplace=True)
matrix = pd.merge(left=matrix, right=group, on=['date_block_num', 'shop_id', 'item_id'], how='left')
matrix = lag_features(matrix, [1,2,3,6,12], 'date_shopitem_avg_item_cnt')
matrix.drop('date_shopitem_avg_item_cnt', axis=1, inplace=True)

# 趋势特征
group = matrix.groupby('item_id').agg({'item_cnt_month': 'mean'})
group.columns = ['trend_item_avg_cnt_month']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on='item_id', how='left')

group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': 'mean'})
group.columns = ['trend_date_item_avg_cnt_month']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['date_block_num', 'item_id'], how='left')

# 计算matrix中商品的历史价格
lags = [1, 2, 3, 4, 5, 6, 12]
matrix = lag_features(matrix, lags, 'trend_date_item_avg_cnt_month')
for i in lags:
    matrix['delta_cnt_month_lag_' + str(i)] = (matrix['trend_date_item_avg_cnt_month_lag_' + str(i)] - matrix[
        'trend_item_avg_cnt_month']) / matrix['trend_item_avg_cnt_month']


def select_trend2(row):
    for i in lags:
        if pd.notnull(row['delta_cnt_month_lag_' + str(i)]):  # 如果不是NaN
            return row['delta_cnt_month_lag_' + str(i)]
    return 0  # 如果delta_price_lag_都为空，那么将趋势设为0，0代表没有趋势


matrix['delta_cnt_month_lag'] = matrix.apply(select_trend2, axis=1)

matrix['delta_cnt_month_lag'] = matrix['delta_cnt_month_lag'].astype(np.float16)

features_to_drop = ['trend_item_avg_cnt_month', 'trend_date_item_avg_cnt_month']
for i in lags:
    features_to_drop += ['trend_date_item_avg_cnt_month_lag_' + str(i)]
    features_to_drop += ['delta_cnt_month_lag_' + str(i)]
matrix.drop(features_to_drop, axis=1, inplace=True)


# 趋势特征 +++++++++++++++++++++++++++++++++++++
group = matrix.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': 'mean'})
group.columns = ['qushi_shop_item_avg_cnt_month']
group = group.reset_index()
matrix = pd.merge(left=matrix, right=group, on=['shop_id', 'item_id'], how='left')

group = matrix.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_month': 'mean'})
group.columns = ['qushi_date_shop_item_avg_cnt_month']
group = group.reset_index()
matrix=pd.merge(left=matrix, right=group, on=['date_block_num', 'shop_id', 'item_id'], how='left')

# 计算matrix中商品的历史价格
lags = [1,2,3,4,5,6,12]
matrix = lag_features(matrix, lags, 'qushi_date_shop_item_avg_cnt_month')
for i in lags:
    matrix['delta2_cnt_month_lag_'+str(i)] = (matrix['qushi_date_shop_item_avg_cnt_month_lag_' + str(i)] - matrix['qushi_shop_item_avg_cnt_month']) / matrix['qushi_shop_item_avg_cnt_month']

def select_trend3(row):
    for i in lags:
        if pd.notnull(row['delta2_cnt_month_lag_'+str(i)]):  # 如果不是NaN
            return row['delta2_cnt_month_lag_'+str(i)]
    return 0   #  如果delta_price_lag_都为空，那么将趋势设为0，0代表没有趋势

matrix['delta2_cnt_month_lag'] = matrix.apply(select_trend3, axis=1)
matrix['delta2_cnt_month_lag'] = matrix['delta2_cnt_month_lag'].astype(np.float16)

features_to_drop = ['qushi_shop_item_avg_cnt_month','qushi_date_shop_item_avg_cnt_month']
for i in lags:
    features_to_drop += ['delta2_cnt_month_lag_'+str(i)]
matrix.drop(features_to_drop, axis=1, inplace=True)

# 因为有12个月的延迟特征（1，2，3，6，12）（1，2，3，4，5，6，12），所以需要删除前12月的数据
matrix = matrix[matrix['date_block_num'] > 11]

# 找到有NaN值的列，然后把那些列中的NaN值填充0
columns = matrix.columns
column_null = []
for i in columns:
    if len(matrix[matrix[i].isnull()]) > 0:
        column_null.append(i)

for i in column_null:
    matrix[i].fillna(0, inplace=True)


"""建模"""

trainData = matrix[matrix['date_block_num'] < 33]
label_train = trainData['item_cnt_month']
X_train = trainData.drop('item_cnt_month', axis=1)

validData = matrix[matrix['date_block_num'] == 33]
label_valid = validData['item_cnt_month']
X_valid = validData.drop('item_cnt_month', axis=1)

import lightgbm as lgb
train_data = lgb.Dataset(data=X_train, label=label_train)
valid_data = lgb.Dataset(data=X_valid, label=label_valid)
params = {
    'objective': 'regression',  # 回归
    'metric': 'rmse',   # 回归问题选择rmse
    'n_estimators': 2000,
    'max_depth': 8,
    'num_leaves': 220,   # 每个弱学习器拥有的叶子的数量
    'learning_rate': 0.005,
    'bagging_fraction': 0.9,    # 每次训练“弱学习器”用的数据比例（应该也是随机的），用于加快训练速度和减小过拟合
    'feature_fraction': 0.3,   # 每次迭代过程中，随机选择30%的特征建树（弱学习器）
    'bagging_seed': 0,
    'early_stop_rounds': 50
}
lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data])
# [2000]	training's rmse: 0.741418	valid_1's rmse: 0.888629   kaggle: 0.93425  34个月(0.92392)
# [10000]	training's rmse: 0.680365	valid_1's rmse: 0.876645    kaggle: 0.92547