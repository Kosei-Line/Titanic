import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn import model_selection
import xgboost as xgb

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

train = pd.read_csv("./train.csv", sep=",")
test = pd.read_csv("./test.csv", sep=",")

data = pd.concat([train, test])
train_size = train.shape[0] # 891
test_size = test.shape[0] # 418

data['Fare'] = data['Fare'].fillna(13.30)
# data = data.assign(
#     Fare = data.apply(lambda f: f.Fare if pd.notnull(f.Fare) else data[data['Pclass'] == f.Pclass]['Fare'].mean(), axis=1)
# )
data['Embarked'] = data['Embarked'].fillna('S')

data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
# pd.crosstab(data['Title'], data['Pclass'])

age_ref = data.groupby('Title').Age.mean()
data = data.assign(
    Age = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
)
del age_ref

data['FamilySize'] = (data['SibSp'] + data['Parch'] + 1).astype(int)
# data[['FamilySize', 'Survived']].groupby(['FamilySize']).agg(['count','mean'])

cols = [
    'Pclass',
    'Age',
    'Sex',
    'FamilySize',
#     'SibSp',
#     'Parch',
    'Fare',
    'Embarked'
]
X_train = data[:train_size][cols]
Y_train = data[:train_size]['Survived'].astype(int)
X_test = data[train_size:][cols]

# print(X_train.shape, Y_train.shape, X_test.shape)

one_hot_features = [
    'Pclass',
    'Sex',    
    'Embarked'
]
X_train = pd.get_dummies(X_train, columns = one_hot_features)
X_test = pd.get_dummies(X_test, columns = one_hot_features)

# print(X_train.shape, Y_train.shape, X_test.shape)

xg_boost = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.65, gamma=2, learning_rate=0.3, max_delta_step=1,
       max_depth=4, min_child_weight=2, missing=None, n_estimators=280,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

xg_boost.fit(X_train, Y_train)
Y_pred = xg_boost.predict(X_test)
print(xg_boost.score(X_train, Y_train))

scores = model_selection.cross_val_score(xg_boost, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on XGBClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

from sklearn.ensemble import GradientBoostingClassifier
forest = GradientBoostingClassifier(n_estimators=55, random_state=8)
forest = forest.fit(X_train, Y_train)
Y_pred_forest = forest.predict(X_test)
print(forest.score(X_train, Y_train))

scores = model_selection.cross_val_score(forest, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on XGBClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

feature_importance = list(zip(X_train.columns.values, xg_boost.feature_importances_))
feature_importance.sort(key=lambda x:x[1])
feature_importance

submission = pd.DataFrame({
    "PassengerId": data[train_size:]["PassengerId"], 
    "Survived": np.logical_and(Y_pred, Y_pred_forest)*1
})
submission.to_csv('submission.csv', index=False)
