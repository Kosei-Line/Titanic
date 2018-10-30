import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda, reporter
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module

class MyUpdater(training.StandardUpdater):
    def __init__(self, iterator, CLS, Opt,
        converter=convert.concat_examples, device=0):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.CLS = CLS
        self._optimizers = {'main':Opt}
        self.converter = converter
        self.device = device
        self.iteration = 0

    def update_core(self):
        iterator = self._iterators['main'].next()
        #入力データ
        input = self.converter(iterator, self.device)
        xp = np if int(self.device) == -1 else cuda.cupy
        x_batch = xp.array(input[0]) #入力データ
        t_batch = xp.array(input[1]) #教師データ
        #lossとaccuracy
        self.loss = 0
        self.acc  = 0
        #計算開始
        y = self.CLS(x_batch, softmax=False)
        self.loss = F.softmax_cross_entropy(y, t_batch)
        self.acc = F.accuracy(y, t_batch)
        #誤差伝播
        self._optimizers['main'].target.cleargrads()
        self.loss.backward()
        self._optimizers['main'].update()
        reporter.report({'main/loss':self.loss, 'main/acc':self.acc})
