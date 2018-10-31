# -*- coding: utf-8 -*-
"""
## 第1章　API周りの初期設定
"""
# はじめにKaggleのアカウントからAPIを使うためにjson持ってきて入れましょう
from google.colab import files
files.upload()

# jsonを入れれたらcolab上で使えるように権限を変えましょう
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls -l /root/.kaggle
!chmod 600 /root/.kaggle/kaggle.json

# これでKaggleのAPIが使えるようになったので、早速 train/test データをダウンロードしましょう
!kaggle competitions download -c titanic



"""
## 第2章　ライブラリの追加
"""
# 今回はNNの定義にkerasを使うのでKerasを入れます。後は予測データを直接kaggleにアップするのでkaggleも入れます。
!pip install kaggle
!pip install keras



"""
## 第3章　実際にやってみる
"""
# 最初にライブラリをインポートしましょう
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.utils import np_utils
from keras.optimizers import SGD

#データをpandasで読み込み、少しだけ表示します。先程ダウンロードしたtrain / test データは content下にあります。
df_train = pd.read_csv("/content/train.csv")
df_test = pd.read_csv("/content/test.csv")
df_train.head(5)

# 12次元のデータの中から、必要そうなラベルだけ取り出します
# 今回はチケットのクラス、性別、年齢、運賃を使用しました(cabinは欠損値多そう)
print('train size: ',df_train.shape,'\nlabel cate:',df_train.columns.values)
features = ['Pclass','Sex','Age','Fare']


# データを整形します。使わないですが、チケットは特に綺麗ではないです
le = LabelEncoder()

df_train["Sex"] = le.fit_transform(df_train["Sex"])
df_test["Sex"] =  le.fit_transform(df_test["Sex"])

df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())
df_test['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_test['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_train['Embarked'] = df_train['Embarked'].fillna("S")
df_test['Embarked'] = df_test['Embarked'].fillna("S")

df_train['Embarked'] = le.fit_transform(df_train['Embarked'])
df_test['Embarked'] = le.fit_transform(df_test['Embarked'])

df_train['Cabin'] = df_train['Cabin'].fillna("None")
df_test['Cabin'] = df_test['Cabin'].fillna("None")
df_train['Cabin'] = le.fit_transform(df_train['Cabin'])
df_test['Cabin'] = le.fit_transform(df_test['Cabin'])

df_train['Ticket'] = le.fit_transform(df_train['Ticket'])
df_test['Ticket'] = le.fit_transform(df_test['Ticket'])


y = df_train['Survived']
x = df_train[features]
x_t = df_test[features]
df_train.head(5)

#train / test データの設定・確認
#今回はデータ数が少ないのでvalidationは無し
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=32)
print("X_train :",X_train.shape)
print("X_test :",X_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)


# モデルの設定
# 今回は畳み込みの方法がわからないので全結合
model = Sequential()

model.add(Dense(64,input_dim=len(features)))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 学習はじめます
# とりあえず1000epoch
y_train_categorical = np_utils.to_categorical(y_train)
history = model.fit(X_train.values, y_train_categorical, nb_epoch=1000)

#acc / lossをグラフにプロットしてみる
acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, '--b', label='loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.clf()

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc,'b', label='accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()

#テストデータに試してみる
y_test_categorical = np_utils.to_categorical(y_test)
loss_and_metrics = model.evaluate(X_test.values, y_test_categorical)
classes = model.predict_classes(x_t.values)

print(loss_and_metrics)




"""
## 第4章　提出してみる
"""
submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": classes})
print(submission[0:10])

submission.to_csv('./titanic_submission.csv', index=False)

# submit the file to kaggle
!kaggle competitions submit titanic -f titanic_submission.csv -m "-Cabin !"
