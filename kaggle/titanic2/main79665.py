import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


# 显示所有列
# pd.set_option('display.max_columns', None)

# 显示所有列
# np.set_printoptions(threshold = np.inf)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv(r'./train.csv')  # 训练数据
test = pd.read_csv(r'./test.csv')  # 测试数据
PassengerId = test['PassengerId']
# 合并训练数据和测试数据
all_data = pd.concat([train, test], ignore_index=True)  # ignore_index=True表示重建索引（行）

# 绘制Survived, Age的密度图
# facet = sns.FacetGrid(train, hue='Survived')
# facet.map(sns.kdeplot, 'Age', shade=True)
# facet.add_legend()
# facet.set(xlim=[0, train['Age'].max()])
# plt.xlabel('年龄')
# plt.ylabel('密度')

# 绘制Embarked和Survived的柱状图
# sns.countplot(x='Embarked', hue='Survived', data=train)

all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

# sns.barplot(x='Title', y='Survived', data=all_data[all_data.Survived.notnull()])
# sns.countplot(x='Title', hue='Survived', data=all_data[all_data.Survived.notnull()])
# plt.xticks(rotation=45)  # 解决xlabel重叠问题

Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))


all_data['Title'] = all_data['Title'].map(Title_Dict)

all_data['FamilySize'] = all_data['Parch'] + all_data['SibSp'] + 1


# 新增FamilyLabel特征
# all_data.loc[(all_data['FamilySize'] >= 2) & (all_data['FamilySize'] <= 4), 'FamilyLabel'] = 2
# all_data.loc[((all_data['FamilySize'] > 4) & (all_data['FamilySize'] <= 7)) | (all_data['FamilySize'] == 1), 'FamilyLabel'] = 1
# all_data.loc[(all_data['FamilySize'] > 7), 'FamilyLabel'] = 0

def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_label)

# 新增Deck特征，将Cabin中为空的数据填充成Unknown,然后提取首字母
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)

"""新增TicketGroup特征"""
Ticket_Count = dict(all_data['Ticket'].value_counts())
# all_data['TicketGroup'] = all_data['Ticket'].map(Ticket_Count)
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])

# 按生存率把TicketGroup分为三类

def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0
all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)

# all_data.loc[(all_data['TicketGroup'] >= 2) & (all_data['TicketGroup'] <= 4), 'TicketGroup'] = 2
# all_data.loc[((all_data['TicketGroup'] > 4) & (all_data['TicketGroup'] <= 8)) | (all_data['TicketGroup'] == 1), 'TicketGroup'] = 1
# all_data.loc[(all_data['TicketGroup'] > 8), 'TicketGroup'] = 0
# sns.barplot(x='TicketGroup', y='Survived', data=all_data)


# 用Pclass,Sex,Title,Parch,SibSp这5个特征构建随机森林填充Age缺失值
from sklearn.ensemble import RandomForestRegressor
# age_df = all_data[['Age', 'Sex', 'Title', 'Parch', 'SibSp', 'Pclass']]
age_df = all_data[['Age', 'Sex', 'Title', 'Pclass']]
age_df = pd.get_dummies(age_df)
# known_age = age_df.loc[age_df.Age.notnull(), :].values
# unknown_age = age_df.loc[age_df.Age.isnull(), :].values
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values
X = known_age[:, 1:]
y = known_age[:, 0]
rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1:])
all_data.loc[all_data.Age.isnull(), 'Age'] = predictedAges  # 填补

# 填补Embarked
# all_data.loc[all_data.Embarked.isnull(), 'Embarked'] = 'S'
all_data['Embarked'] = all_data['Embarked'].fillna('C')

# 填补Fare
# all_data.loc[all_data.Fare.isnull(), 'Fare'] = all_data.loc[(all_data.Pclass == 3) & (all_data.Embarked == 'S'), 'Fare'].mean()
fare = all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.mean()
all_data['Fare']=all_data['Fare'].fillna(fare)


# 新增Surname特征
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].map(Surname_Count)
Female_Child_Group = all_data.loc[(all_data.FamilyGroup >= 2) & ((all_data.Sex == 'female') | (all_data.Age <= 12)), :]  # 妇女儿童组

Male_Adult_Group = all_data.loc[(all_data.FamilyGroup >= 2) & (all_data.Sex == 'male') & (all_data.Age > 12), :]  # 男性成年组

a = Female_Child_Group.groupby('Surname')['Survived'].mean()  # Series对象
Dead_List = set(a.loc[a.apply(lambda x: x == 0)].index)

a = Male_Adult_Group.groupby('Surname')['Survived'].mean()  # Series对象
Survived_List = set(a.loc[a.apply(lambda x: x == 1)].index)

# 对测试数据中的Sex,Age,Title进行处罚修改
train = all_data.loc[all_data.Survived.notnull()]
test = all_data.loc[all_data.Survived.isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Title'] = 'Mr'

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)), 'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)), 'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)), 'Title'] = 'Miss'

# 划分训练，测试数据集
all_data = pd.concat([train, test])
all_data = all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data = pd.get_dummies(all_data)   # 只转化非数值的特征
train = all_data[all_data.Survived.notnull()]
test = all_data[all_data.Survived.isnull()].drop('Survived', axis=1)  # all_data[all_data.Survived.isnull()]对这个数据不会改变
X = train.values[:, 1:]  # (891, 25)
y = train.values[:, 0]  # (891,)


"""随机森林"""
# 建模优化
pipe = Pipeline([
    ('select', SelectKBest(k=20)),
    ('classify', RandomForestRegressor(random_state=10, max_features='sqrt'))  # max_features='sqrt'表示每棵树能够使用总特征数的平方根个数量
])
param_grid = {
    'classify__n_estimators': list(range(20, 50, 2)),
    'classify__max_depth': list(range(3, 60, 3))
}

# 评估模型
select = SelectKBest(k=20)
clf = RandomForestClassifier(
    n_estimators=26,
    max_depth=6,
    max_features='sqrt',
    random_state=10,
    warm_start=True,
)
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)
cv_score = cross_val_score(pipeline, X, y, cv=10)
predictions_forest = pipeline.predict(test)


"""SVC"""
pipe = Pipeline([
    ('select', SelectKBest(k=20)),
    ('std_scaler', StandardScaler()),
    ('classify', SVC(kernel='rbf'))
])
param_grid = {
    'classify__gamma': [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05],
}
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc')
# gsearch.fit(X, y)
# print(gsearch.best_params_, gsearch.best_score_)

pipeline = Pipeline([
    ('select', SelectKBest(k=20)),
    ('std_scaler', StandardScaler()),
    ('classify', SVC(kernel='rbf', gamma=0.07))
])
pipeline.fit(X, y)
cv_score = cross_val_score(pipeline, X, y, cv=10)
# print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
predictions_svc = pipeline.predict(test)


"""Logistic"""
pipe = Pipeline([
    ('select', SelectKBest(k=20)),
    ('poly', PolynomialFeatures()),
    ('std_scaler', StandardScaler()),
    ('classify', LogisticRegression(penalty='l2'))
])
param_grid = {
    'poly__degree': list(range(1, 5)),
    'classify__C': list(np.linspace(0.1, 2, 20)),
}
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
# gsearch.fit(X, y)
# print(gsearch.best_params_, gsearch.best_score_)

pipeline = Pipeline([
    ('select', SelectKBest(k=20)),
    ('poly', PolynomialFeatures(degree=4)),
    ('std_scaler', StandardScaler()),
    ('classify', LogisticRegression(penalty='l1', C=0.05))
])
pipeline.fit(X, y)
cv_score = cross_val_score(pipeline, X, y, cv=10)
predictions_log = pipeline.predict(test)




# 标准化
# test = test.values
# std_scaler = StandardScaler()
# std_scaler.fit(test)
# test = std_scaler.transform(test)

# 预测
a = (predictions_forest + predictions_svc + predictions_log) >= 2
predictions = np.array(a, dtype='int')
# predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv('submit/submit_df_ensemble2.csv', index=False)




plt.show()
