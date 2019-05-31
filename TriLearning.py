import sklearn
from math import floor
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim

# Adapted for Pytorch from: https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning

class TriTraining:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        if len(self.classifiers) != 3:
            print("Not 3 Classifiers")

    def fit(self, labeled_x, labeled_y, unlabeled_x):
        #Initialize
        self.thirds = [None]*3
        for i in range(3):
            print("initializing classifier ", i)
            self.thirds[i] = Third(self.classifiers[i], labeled_x, labeled_y)

        third_rotations = [rotate(self.thirds, i) for (i, _) in enumerate(self.thirds)]

        print("start training with unlabeled data")
        changed = True
        i = 0
        while changed:
            i += 1
            print("iteration number: ", i)
            changed = False
            for t1, t2, t3 in third_rotations:
                changed |= t1.train(t2, t3, unlabeled_x)


    def predict(self, X):
        predictions = np.asarray([third.predict(X) for third in self.thirds])
        import scipy
        return scipy.stats.mstats.mode(predictions).mode[0]


    def score(self, X, y_true):
        y_true = y_true.astype(int)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

class Third:
    def __init__(self, classifier, labeled_X, labeled_y):
        self.classifier = classifier
        self.labeled_X = labeled_X
        self.labeled_y = labeled_y

        sample = sklearn.utils.resample(self.labeled_X, self.labeled_y)  # BootstrapSample(L)

        self.train_classifier(sample[0], sample[1])

        self.err_prime = 0.5  # e'i = 0.5
        self.l_prime = 0.0  # l'i = 0.0

    def update(self, L_X, L_y, error):
        X = np.append(self.labeled_X, L_X, axis=0)
        y = np.append(self.labeled_y, L_y, axis=0)
        self.train_classifier(X, y)
        self.err_prime = error
        self.l_prime = len(L_X)

    def train_classifier(self, X, y):
        dataset = data_utils.TensorDataset(torch.Tensor(X), torch.Tensor(y))
        dataloader = data_utils.DataLoader(dataset, batch_size=64)
        #traning the Network on labeled data
        lr = 0.001
        opt = optim.Adam(self.classifier.parameters(), lr=lr)
        loss_fun = nn.CrossEntropyLoss()
        num_epochs = 100

        self.classifier.train()
        for epoch in range(num_epochs):
            if epoch % 10 == 0:
                lr = lr / 2
                for g in opt.param_groups:
                    g['lr'] = lr

            for batch_id, (data, target) in enumerate(dataloader):
                output = self.classifier(data)
                loss = loss_fun(output, target.long())
                opt.zero_grad()
                loss.backward()
                opt.step()

    def train(self, t1, t2, unlabeled_X):
        L_X = []
        L_y = []
        error = self.measure_error(t1, t2)
        print(error)
        if (error >= self.err_prime):
            print("Code: 1")
            return False

        for X in unlabeled_X:
            X = X.reshape(1, -1)
            y = t1.predict(X)
            if y == t2.predict(X):
                L_X.append(X)
                L_y.append(y)

        count_of_added = len(L_X)
        # Turn the python list of chosen samples into numpy array
        L_X = np.concatenate(L_X)
        L_y = np.concatenate(L_y)

        if (self.l_prime == 0):
            self.l_prime = floor(error / (self.err_prime - error) + 1)

        if self.l_prime >= count_of_added:
            print("Code: 2")
            return False
        if error * count_of_added < self.err_prime * self.l_prime:
            self.update(L_X, L_y, error)
            print("Code: 3")
            return True
        if self.l_prime > error / (self.err_prime - error):
            n = floor(self.err_prime * self.l_prime / error - 1)
            L_X, L_y = sklearn.utils.resample(L_X, L_y, replace=False, n_samples=n)

            self.update(L_X, L_y, error)
            return True
        print("Code: 4")
        return False

    def measure_error(self, third_1, third_2):
        prediction_1 = third_1.predict(self.labeled_X)
        prediction_2 = third_2.predict(self.labeled_X)
        both_incorrect = np.count_nonzero((prediction_1 != self.labeled_y) & (prediction_2 != self.labeled_y))
        both_same = np.count_nonzero(prediction_1 == prediction_2)
        if(both_same == 0): return np.inf
        error = both_incorrect/both_same
        return error

    def predict(self, X):
        self.classifier.eval()
        with torch.no_grad():
            output = self.classifier(torch.Tensor(X)).max(-1)[1]
        self.classifier.train()
        return output.numpy()


#Helper for rotating a list
def rotate(l, n):
    return l[n:] + l[:n]