import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel(r'C:\Users\pashu\OneDrive\Desktop\learning space\Training data.xlsx')

x_train = data.iloc[:, 0:8]
y_train = data.iloc[:, 8].values.reshape(-1, 1)


def feature_changing(x_train):
    x_train.iloc[:, 0] = np.where(x_train.iloc[:, 0] == 'yes', 1, 0)
    x_train.iloc[:, 1] = np.where(x_train.iloc[:, 1] == 'M', 1, 0)
    return x_train


x_train=feature_changing(x_train)

for i in range(x_train.shape[1]):
    plt.scatter(x_train.iloc[:, i], y_train, label=f'Feature {i}')
    plt.xlabel(f'Feature {i}')
    plt.ylabel('y_train')
    plt.title(f'Feature {i} vs y_train')
    plt.legend()
    plt.show()



def z_score(x_train):
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    z_scores = (x_train - x_mean) / x_std
    return z_scores, x_mean, x_std

def cost(x_train, y_train, w, b):
    y_pred = np.dot(x_train, w) + b
    m = x_train.shape[0]
    loss = (1 / m) * np.sum(np.power((y_train - y_pred), 2))
    return loss

def gradient_descent(x_train, y_train, w, b, learning_rate=0.1):
    m = x_train.shape[0]
    y_pred = np.dot(x_train, w) + b
    dw = (-2 / m) * np.dot(x_train.T, (y_train - y_pred))
    db = (-2 / m) * np.sum(y_train - y_pred)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

x_train, x_mean, x_std = z_score(x_train)

np.random.seed(2147483647)
w = np.random.randn(x_train.shape[1], 1)
b = np.random.randn(1)

old_cost = cost(x_train, y_train, w, b)
while True:
    w, b = gradient_descent(x_train, y_train, w, b)
    new_cost = cost(x_train, y_train, w, b)
    if abs(old_cost - new_cost) < 0.00001:
        break
    old_cost = new_cost

test_data = pd.read_excel(r'C:\Users\pashu\OneDrive\Desktop\learning space\Test data.xlsx')
x_predict = test_data.iloc[:, :8]





x_predict = feature_changing(x_predict)
x_predict = (x_predict - x_mean) / x_std
ans = test_data.iloc[:, 8].to_numpy()

y_predict = np.dot(x_predict, w) + b

accuracy = np.mean(np.abs(y_predict.flatten() - ans) < 0.5) * 100
ok = 'Congratulations' if accuracy > 95 else 'Optimization required'
print(f"{ok}, your accuracy is {accuracy:.2f}%")
