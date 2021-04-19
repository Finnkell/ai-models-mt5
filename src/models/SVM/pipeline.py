import random


class ScrapyTree():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for _row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)

        return predictions

