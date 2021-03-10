# 选择重要特征
fi = lgb_model.feature_importance()
sorted_index = np.argsort(-fi)
sorted_index = sorted_index[:60]
sel_cols = list(X_train.columns[sorted_index]) + ['item_cnt_month']
matrix_sel_col = matrix[sel_cols]


# PCA
from sklearn.decomposition import PCA
pca_matrix = matrix.copy()
pca_matrix = pca_matrix.drop('item_cnt_month', axis=1)
pca = PCA(n_components=50)
pca.fit(pca_matrix)

"""建模"""
trainData = matrix[matrix['date_block_num'] < 33]
label_train = trainData['item_cnt_month']
X_train = trainData.drop('item_cnt_month', axis=1)

validData = matrix[matrix['date_block_num'] == 33]
label_valid = validData['item_cnt_month']
X_valid = validData.drop('item_cnt_month', axis=1)

X_train_transform = pca.transform(X_train)
X_valid_transform = pca.transform(X_valid)


# 对sub数据进行变换
def myfun(val):
    return round(val, 0)
sub['item_cnt_month'] = sub['item_cnt_month'].apply(myfun)


# 验证
from sklearn.metrics import mean_squared_error
valid_prediction = lgb_model.predict(X_valid).clip(0,20)
rmse_valid = np.sqrt(mean_squared_error(valid_prediction, label_valid))
