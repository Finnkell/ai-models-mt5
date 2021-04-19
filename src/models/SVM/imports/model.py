from sklearn import svm
from sklearn import metrics
import sklearn.preprocessing as skp
import numpy as np

class SVM():

    def __init__(self):
        self.model_results = []
        self.svm_model = None
        self.model_predict = None
        self.n_candles = 4
        self.scaler = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.model_predict = None


    def build_svm(self, C_svm=1.0, kernel_svm='linear', degree_svm=2, gamma_svm=1, size_svm=1000, n_candles_svm=4):
        
        self.scaler = skp.StandardScaler(with_mean=True, with_std=True)

        self.svm_model = svm.SVC(
            C=C_svm,
            kernel=kernel_svm,
            degree=degree_svm,
            gamma=gamma_svm,
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=6000,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=None
        )


    def split_train_test_data(self, dataset, size):
        close = dataset['Close']
        n = self.n_candles

        X = []
        Y = []

        for _i in range(len(close) - (n + 1)):
            lista = []

            for _j in range(_i, _i + (n)):
                lista.append(close[_j + 1] - close[_j])

            if (close[_i + n + 1] - close[_i + n]) > 0:
                Y.append(1)
            elif (close[_i + n + 1] - close[_i + n]) < 0:
                Y.append(0)
            else:
                Y.append(1)

            X.append(lista)

        X_svm = X[len(X) - size:]
        Y_svm = Y[len(Y) - size:]

        prop = int(len(X_svm)*0.8)

        print(f'\n\n prop = {prop}')

        self.x_train = np.array(X_svm[:prop])
        self.x_test  = np.array(X_svm[prop:])
        self.y_train = np.array(Y_svm[:prop])
        self.y_test  = np.array(Y_svm[prop:])


    def train_model(self):

        self.scaler.fit(self.x_train)

        x_train_tr = self.scaler.transform(self.x_train)
        x_test_tr = self.scaler.transform(self.x_test)

        self.svm_model.fit(X=x_train_tr, y=self.y_train, sample_weight=None)

        self.model_predict = self.svm_model.predict(x_test_tr)


    def print_results(self):
        accuracy = metrics.accuracy_score(self.y_test, self.model_predict)
        precision = metrics.precision_score(self.y_test, self.model_predict)
        recall = metrics.recall_score(self.y_test, self.model_predict)
        t1_score = (2*metrics.precision_score(self.y_test, self.model_predict)*metrics.recall_score(self.y_test, self.model_predict))/(metrics.precision_score(self.y_test, self.model_predict) + metrics.recall_score(self.y_test, self.model_predict))
        auc_score = metrics.roc_auc_score(self.y_test, self.model_predict)
        kappa = metrics.cohen_kappa_score(self.y_test, self.model_predict)

        self.model_results.append(accuracy)
        self.model_results.append(precision)
        self.model_results.append(recall)
        self.model_results.append(t1_score)
        self.model_results.append(auc_score)
        self.model_results.append(kappa)

        print(self.model_results)


    def save_results(self, file_name):
        pass