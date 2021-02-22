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
from sklearn.ensemble import VotingClassifier


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
all_data.loc[(all_data['TicketGroup'] >= 2) & (all_data['TicketGroup'] <= 4), 'TicketGroup'] = 2
all_data.loc[((all_data['TicketGroup'] > 4) & (all_data['TicketGroup'] <= 8)) | (all_data['TicketGroup'] == 1), 'TicketGroup'] = 1
all_data.loc[(all_data['TicketGroup'] > 8), 'TicketGroup'] = 0
# sns.barplot(x='TicketGroup', y='Survived', data=all_data)

# 用Pclass,Sex,Title三个特征进行分组后年龄的中值填补Age缺失值
group_for_age = all_data.groupby(by=['Pclass', 'Sex', 'Title'])['Age'].mean()   # Series对象 'Pclass', 'Sex', 'Title'是索引
group_for_age = group_for_age.reset_index()[['Pclass', 'Sex', 'Title', 'Age']]
all_data.loc[all_data.Age.isnull(), 'Age'] = all_data.loc[all_data.Age.isnull()].apply(lambda x: group_for_age.loc[(group_for_age.Sex == x.Sex) & (group_for_age.Title == x.Title) & (group_for_age.Pclass == x.Pclass), 'Age'].values[0], axis=1)   # axis=1表示取每一行的数据

# 用Pclass,Sex,Title,Parch,SibSp这5个特征构建随机森林填充Age缺失值
# age_df = all_data[['Age', 'Sex', 'Title', 'Parch', 'SibSp', 'Pclass']]
# age_df = all_data[['Age', 'Sex', 'Title', 'Pclass']]
# age_df = pd.get_dummies(age_df)
# known_age = age_df[age_df.Age.notnull()].values
# unknown_age = age_df[age_df.Age.isnull()].values
# X = known_age[:, 1:]
# y = known_age[:, 0]
# rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
# rfr.fit(X, y)
# predictedAges = rfr.predict(unknown_age[:, 1:])
# all_data.loc[all_data.Age.isnull(), 'Age'] = predictedAges  # 填补

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

# 新增SurnameGroupLabel特征
def sur_group_label(s):
    if s >= 2 and s <= 3:
        return 2
    elif (s > 3 and s <= 8) or s == 1:
        return 1
    elif s > 8:
        return 0
all_data['SurnameGroupLabel'] = all_data['FamilyGroup'].apply(lambda row: sur_group_label(row))

################### trick
# 在家庭人数大于等于2的组中，如果至少有一名男士获救，则其它组成员获救的可能性极大
# a = all_data[(all_data.Survived == 1) & (all_data.FamilySize >= 2) & (all_data.Sex == 'male')].Surname.values
# a = set(a)
# all_data.loc[(all_data.Survived.isnull()) & (all_data.Surname.isin(a)), 'Sex'] = 'female'
# all_data.loc[(all_data.Survived.isnull()) & (all_data.Surname.isin(a)), 'Age'] = 5
# all_data.loc[(all_data.Survived.isnull()) & (all_data.Surname.isin(a)), 'Title'] = 'Miss'
#
# # 在家庭人数大于等于2的组中，如果至少有一名女士遇难，则其它组成员获救的可能性极小
# a = all_data[(all_data.Survived == 0) & (all_data.FamilySize >= 2) & (all_data.Sex == 'female')].Surname.values
# a = set(a)
# all_data.loc[(all_data.Survived.isnull()) & (all_data.Surname.isin(a)), 'Sex'] = 'male'
# all_data.loc[(all_data.Survived.isnull()) & (all_data.Surname.isin(a)), 'Age'] = 60
# all_data.loc[(all_data.Survived.isnull()) & (all_data.Surname.isin(a)), 'Title'] = 'Mr'
################### trick  end

Female_Child_Group = all_data.loc[(all_data.FamilyGroup >= 2) & ((all_data.Sex == 'female') | (all_data.Age <= 12)), :]  # 妇女儿童组
Male_Adult_Group = all_data.loc[(all_data.FamilyGroup >= 2) & (all_data.Sex == 'male') & (all_data.Age > 12), :]  # 男性成年组

a = Female_Child_Group.groupby('Surname')['Survived'].mean()  # Series对象 Surname是索引
# Dead_List = set(a.loc[a.apply(lambda x: x == 0)].index)
Dead_List = set(a.loc[a.apply(lambda x: x < 1)].index)

a = Male_Adult_Group.groupby('Surname')['Survived'].mean()  # Series对象  Surname是索引
# Survived_List = set(a.loc[a.apply(lambda x: x == 1)].index)
Survived_List = set(a.loc[a.apply(lambda x: x > 0)].index)

# 对测试数据中的Sex,Age,Title进行处罚修改，加上惩罚后，准确率会略微提高
train = all_data.loc[all_data.Survived.notnull()]
test = all_data.loc[all_data.Survived.isnull()]
test.loc[(test['Surname'].isin(Dead_List)), 'Sex'] = 'male'
test.loc[(test['Surname'].isin(Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].isin(Dead_List)), 'Title'] = 'Mr'

test.loc[(test['Surname'].isin(Survived_List)), 'Sex'] = 'female'
test.loc[(test['Surname'].isin(Survived_List)), 'Age'] = 5
test.loc[(test['Surname'].isin(Survived_List)), 'Title'] = 'Miss'


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

# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc')
# gsearch.fit(X, y)
# print(gsearch.best_params_, gsearch.best_score_)

# 评估模型
select = SelectKBest(k=20)
clf = RandomForestClassifier(
    n_estimators=26,
    max_depth=6,
    max_features='sqrt',
    random_state=10,
    warm_start=True,
)
pipeline_forest = make_pipeline(select, clf)
# pipeline_forest.fit(X, y)
# cv_score = cross_val_score(pipeline_forest, X, y, cv=10)
# print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))


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

pipeline_svc = Pipeline([
    ('select', SelectKBest(k=20)),
    ('std_scaler', StandardScaler()),
    ('classify', SVC(kernel='rbf', gamma=0.08, probability=True))
])
# pipeline_svc.fit(X, y)
# cv_score = cross_val_score(pipeline_svc, X, y, cv=10)
# print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
# predictions_svc = pipeline_svc.predict(test)


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

pipeline_log = Pipeline([
    ('select', SelectKBest(k=20)),
    ('poly', PolynomialFeatures(degree=3)),
    ('std_scaler', StandardScaler()),
    ('classify', LogisticRegression(penalty='l2', C=0.01))
])
# pipeline_log.fit(X, y)
# cv_score = cross_val_score(pipeline_log, X, y, cv=10)
# print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
# predictions_log = pipeline_log.predict(test)


"""Gradient Boosting """
from sklearn.ensemble import GradientBoostingClassifier
pipe = Pipeline([
    ('select', SelectKBest(k=20)),
    ('classify', GradientBoostingClassifier())  # max_features='sqrt'表示每棵树能够使用总特征数的平方根个数量
])
param_grid = {
      'classify__loss': ["deviance"],
      'classify__n_estimators': [100,200,300],
      'classify__learning_rate': [0.1, 0.05, 0.01],
      'classify__max_depth': [4, 8],
      'classify__min_samples_leaf': [100,150],
      'classify__max_features': [0.3, 0.1]
}
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc')
# gsearch.fit(X, y)
# print(gsearch.best_params_, gsearch.best_score_)
pipeline_gb = Pipeline([
    ('select', SelectKBest(k=20)),
    ('classify', GradientBoostingClassifier(
        loss='deviance',
        n_estimators=400,
        learning_rate=0.1,  # 步长
        max_depth=8,
        min_samples_leaf=100,
        max_features=0.3
    ))
])
pipeline_gb.fit(X, y)
cv_score = cross_val_score(pipeline_gb, X, y, cv=10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
exit()


"""集成学习"""
voting_clf = VotingClassifier(
    estimators=[
        ('forest', pipeline_forest),
        ('svc', pipeline_svc),
        # ('log', pipeline_log),
        ('gb', pipeline_gb)
    ],
    voting='hard'
)
voting_clf.fit(X, y)
cv_score = cross_val_score(voting_clf, X, y, cv=10)
# print(type(cv_score))
# print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

# 标准化
# test = test.values
# std_scaler = StandardScaler()
# std_scaler.fit(test)
# test = std_scaler.transform(test)

# 预测
# a = (predictions_forest + predictions_svc + predictions_log) >= 2
# predictions = np.array(a, dtype='int')
predictions = voting_clf.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv('submit/submit_df.csv', index=False)



plt.show()
