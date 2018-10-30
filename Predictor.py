import chainer, os
import pandas as pd
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import datasets, iterators, optimizers, serializers, training
from chainer.training import extensions

import Mod
args = Mod.args
Net = Mod.Net

def Load_Dataset():
    #csvからデータを読み取る
    data = pd.read_csv('test.csv')
    #必要なデータだけ取り出す
    data = data[['Pclass','Sex', 'Age', 'Fare', 'Parch', 'Embarked',
     'SibSp']]
    #NaNをなくす
    data['Age'] = data['Age'].fillna(method='ffill')#1つ前のをNaNにコピー
    #data['Fare'] = round(data['Fare'], -1)
    data['Age'] = round(data['Age'], -1) #10の位に統一
    data["Embarked"] = data["Embarked"].fillna("S")
    #2値に変える
    data['Sex'] = data['Sex'].replace('male', 0)
    data['Sex'] = data['Sex'].replace('female', 1)
    data["Embarked"] = data["Embarked"].replace("S", 0)
    data["Embarked"] = data["Embarked"].replace("C", 1)
    data["Embarked"] = data["Embarked"].replace("Q", 2)
    data['Parch'] = data['Parch'] + data['SibSp']
    #データを行列に変換
    data = data.as_matrix()
    #入力データX，教師データT
    X = data[:, :-1].astype('float32')
    #テストデータ
    test  = X
    #テストデータを返す
    return test

def main():
        #学習ネットワークを持ってくる
    CLS = Net.CLS()
    #gpuを使う
    #CLS.to_gpu()
    #データセットの読み込み
    print('Loading dataset')
    test = Load_Dataset()
    print('Loaded dataset')

    a = []
    b = []
    print(test[[0]])
    serializers.load_npz('result/b{}/CLS_epoch_{}'.format(args.batch,
        args.epoch), CLS)
    for i in range(len(test)):
        y = CLS(test[[i]]).data.argmax(axis=1)[0]
        a.append(892+i)
        b.append(y)
    df = pd.DataFrame({
        'PassengerId' : a,
        'Survived' : b
    })
    df.to_csv("submit.csv",index=False)

if __name__ == '__main__':
    main()
