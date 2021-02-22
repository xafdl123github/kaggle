import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练数据
train = pd.read_csv('./data/sales_train.csv')   # (2935849, 6)

# 测试数据
test = pd.read_csv('./data/test.csv')   # (214200, 3)

# 分析商品类别 item_categories添加 split type sub_type type_code sub_type_code 特征
itemcat = pd.read_csv('./data/item_categories.csv')
itemcat['split'] = itemcat.item_category_name.str.split('-')
itemcat['type'] = itemcat['split'].map(lambda x:x[0])
itemcat['sub_type'] = itemcat['split'].map(lambda x:x[1] if len(x) > 1 else x[0])
le = LabelEncoder()
itemcat['type_code'] = le.fit_transform(itemcat.type)
itemcat['sub_type_code'] = le.fit_transform(itemcat.sub_type)

# 合并[train,test],items
items = pd.read_csv('./data/items.csv')
items = items[['item_id', 'item_category_id']]
train = pd.merge(left=train, right=items, on='item_id')
# 对原数据没有改变
test = pd.merge(left=test, right=items, on='item_id')
test = test.sort_values(by='ID')
test = test.reset_index(drop=True)

# 删除训练数据中的异常数据
train = train[train.item_cnt_day <= 1000]
train = train[train.item_price <= 100000]

# 中位数填补item_price为负数的数据
price_df = train[train.item_id == 2973]
train.loc[train.item_price < 0, 'item_price'] = price_df.item_price.median()

# 构造item_cnt_month特征
train2 = train.groupby(['shop_id','item_id','date_block_num']).agg({'item_cnt_day': 'sum'}).reset_index()
train2 = train2.rename(columns={'item_cnt_day': 'item_cnt_month'})

# date_block到month的映射
month_lst = list(range(1, 13)) + list(range(1, 13)) + list(range(1, 11))
date_num_lst = list(range(0, 34))
dateBlock2month = dict(tuple(zip(date_num_lst, month_lst)))

# train2添加month特征
train2['month'] = train2['date_block_num'].map(dateBlock2month)

# test添加date_block_num，month特征
test['date_block_num'] = 34
test['month'] = 11

# train2添加item_category_id特征
train2 = pd.merge(left=train2, right=items, on='item_id', how='left')

# train2添加type_code,sub_type_code
itemcat_simple = itemcat[['item_category_id','type_code','sub_type_code']]
train2 = pd.merge(left=train2, right=itemcat_simple, on='item_category_id', how='left')

# test添加type_code,sub_type_code
test = pd.merge(left=test, right=itemcat_simple, on='item_category_id', how='left')

# 分析shops
shops = pd.read_csv('./data/shops.csv')
shops['split']=shops.shop_name.str.split(' ')
shops['shop_city'] = shops['split'].map(lambda x:x[0])
shops['shop_city_code'] = le.fit_transform(shops['shop_city'])

# train2,test添加shop_city_code特征
shops_simple = shops[['shop_id','shop_city_code']]
train2 = pd.merge(left=train2, right=shops_simple, on='shop_id', how='left')
test = pd.merge(left=test, right=shops_simple, on='shop_id', how='left')

# 合并train2,test
# all_data = pd.concat([train2,test], axis=0, ignore_index=True)  # 全部数据

# X = train2.drop(['item_cnt_month','item_category_id'], axis=1)
# target = train2['item_cnt_month']