import chainer
import chainer.functions as F
import chainer.links as L

class CLS(chainer.Chain):
    def __init__(self):
        super(CLS, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(6, 128)
            self.fc2 = L.Linear(None, 128)
            self.fc3 = L.Linear(None, 128)
            self.fc4 = L.Linear(None, 2)
    def __call__(self, x, softmax=False):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        #h = F.dropout(h)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        if softmax:
            return F.softmax(h)
        else:
            return h