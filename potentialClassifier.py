import numpy as np
from sklearn import metrics
# Написать ядро для уменьшения нулевых потенциалов. 
# Уменьшить количество нулевых объектов(в фиолетовом классе)

class PotentialClassifier():
    def __init__(self, train_x, train_y, kernel, window_width, epoch_number) -> None:
        self.classes = np.unique(train_y) #  получаем классы
        self.train_x = train_x # переобозначеный указатель
        self.train_y = train_y # переобозначеный указатель
        self.charges = np.zeros_like(train_y) # массив параметров, задающих "заряд", т.е. степень важности объекта при классификации
        self.indexes = np.arange(0, len(train_y)) # индексы в массиве классов
        self.Kernel = kernel # функция, убывающая с ростом аргумента.
        self.window_width = window_width # параметр, задающий "ширину потенциала"
        self.epoch_number = epoch_number # количество эпох


    def minkowski_distances(self, u, v, p=2):
        return np.sum(((u - v)**p), -1)**(1/p)


    def predict(self, x: np.array):
        test_x = np.copy(x)

        if len(test_x.shape) < 2:
            test_x = test_x[np.newaxis, :]
        u = test_x[:, np.newaxis, :]
        v = self.train_x[np.newaxis, :, :]
        weights = self.charges * self.Kernel(self.minkowski_distances(u, v) / self.window_width)
        table = np.zeros((test_x.shape[0], len(self.classes)))
        for class_ in self.classes:
            table[:, class_] = np.sum(weights[:, self.train_y == class_], axis = 1)
        return np.argmax(table, axis = 1)


    def fit(self):
        self.charges[0] = 1
        for _ in range(self.epoch_number):
            for i in range(self.train_x.shape[0]):
                if self.predict(self.train_x[i]) != self.train_y[i]:
                    self.charges[i] += 1
        # удаление всех сэмплов с нулевым "зарядом"
        non_zero_mask = self.charges != 0
        self.train_x = self.train_x[non_zero_mask, ...]
        self.train_y = self.train_y[non_zero_mask, ...]
        self.charges = self.charges[non_zero_mask, ...]
        self.indexes = self.indexes[non_zero_mask, ...]


    def show_accuracy(self, X, y, test_x, test_y):
        predict_arr = self.predict(test_x)
        print("Accuracy")
        print("On test  = ", metrics.accuracy_score(test_y, predict_arr))
        print("On train = ", metrics.accuracy_score(self.train_y, self.predict(self.train_x)))
        print("On full data: ", metrics.accuracy_score(y, self.predict(X)))


    def set_epoch_num(self, epoch_number):
        self.epoch_number = epoch_number


    def get_bad_prediction_arr(self, test_x, test_y):
        bad_predictions_array = list()
        predict_arr = self.predict(test_x)
        for i in range(len(test_y)):
            if predict_arr[i] != test_y[i]:
                bad_predictions_array.append(i)
        return bad_predictions_array