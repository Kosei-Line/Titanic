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
Evaluator = Mod.Evaluator
Updater = Mod.Updater

def Load_Dataset():
    #csvからデータを読み取る
    data = pd.read_csv('train.csv')
    #必要なデータだけ取り出す
    data = data[['Pclass','Sex', 'Age', 'Fare', 'Parch', 'Embarked',
     'SibSp', 'Survived']]
    #NaNをなくす
    #data = data.dropna() #NaNがあったらデータを消す処理
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
    X = data[:,:-2].astype('float32')
    T = data[:,-1:].flatten().astype('int32')
    #訓練データとテストデータに分ける
    thresh_hold = int(X.shape[0]*0.8)
    train = datasets.TupleDataset(X[:thresh_hold], T[:thresh_hold])
    test  = datasets.TupleDataset(X[thresh_hold:], T[thresh_hold:])
    #訓練データとテストデータを返す
    return train, test

def main():
    #学習ネットワークを持ってくる
    CLS = Net.CLS()
    #gpuを使う
    CLS.to_gpu()
    #データセットの読み込み
    print('Loading dataset')
    train, test = Load_Dataset()
    print('Loaded dataset')

    #make_optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.9, beta2=0.999):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer
    Opt = make_optimizer(CLS)
    #set iterator
    train_iter = iterators.SerialIterator(train, args.batch)
    test_iter  = iterators.SerialIterator(test, args.batch,
        repeat=False, shuffle=False)
    #define updater
    updater = Updater.MyUpdater(train_iter, CLS, Opt, device=args.gpu)
    #define trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
        out="{}/b{}".format(args.out, args.batch))
    #define evaluator
    trainer.extend(Evaluator.MyEvaluator(test_iter, CLS, device=args.gpu))
    #save model
    trainer.extend(extensions.snapshot_object(CLS,
        filename='CLS_epoch_{.updater.epoch}'),
        trigger=(args.snapshot, 'epoch'))
    #out Log
    trainer.extend(extensions.LogReport())
    #print Report
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/acc', 'val/loss',
         'val/acc', 'elapsed_time']))
    #display Progress bar
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    del trainer


if __name__ == '__main__':
    main()
