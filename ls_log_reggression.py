import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r'C:\Users\pashu\OneDrive\Desktop\rand_yio\data.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
X_train = X
y_train = y


def plot_data(X, y, positive_label="y=1", negative_label="y=0"):
    positive = y == 1
    negative = y == 0
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=positive_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=negative_label)


plot_data(X_train, y_train[:], positive_label="Accepted", negative_label="Rejected")

plt.xlabel('x_train')
plt.ylabel('y_train')
plt.legend(loc="upper right")
plt.show()

def map_feature(X1, X2):
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

print("Original shape of data:", X_train.shape)
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", X_mapped.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_cost(X, y, w, b, lambda_=1):
    m = X.shape[0]
    y_pred = sigmoid(np.dot(X, w) + b)
    loss = (1 / (2 * m)) * np.sum(np.power((y_pred - y), 2))
    return loss


def compute_cost_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    y_pred = sigmoid(np.dot(X, w) + b)
    loss = (1 / (2 * m)) * np.sum(np.power((y_pred - y), 2))
    cost_reg = (lambda_ / (2 * m)) * np.sum(np.square(w))
    total_cost = cost_reg + loss
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    y_pred = sigmoid(np.dot(X, w) + b)
    error = y_pred - y
    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)
    return dj_db, dj_dw


def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    y_pred = sigmoid(np.dot(X, w) + b)
    error = y_pred - y
    dj_dw = (1 / m) * np.dot(X.T, error) + (lambda_ / m) * w
    dj_db = (1 / m) * np.sum(error)
    return dj_db, dj_dw


def gradient_descent(X, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, lambda_):
    w = initial_w
    b = initial_b
    J_history = []
    
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(X, y, w, b, lambda_)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b, lambda_)
            J_history.append(cost)
            print(f"Iteration {i}: Cost {cost}")

    return w, b, J_history


np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0
lambda_ = 0.01
iterations = 10000
alpha = 0.05


w, b, J_history = gradient_descent(X_mapped, y_train, initial_w, initial_b, compute_cost_reg, compute_gradient_reg, alpha, iterations, lambda_)


def plot_decision_boundary(w, b, X, y):
    plot_data(X[:, 0:2], y)
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        plt.plot(plot_x, plot_y, c="b")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)
        z = z.T
        plt.contour(u, v, z, levels=[0.5], colors="g")

plot_decision_boundary(w, b, X_mapped, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.legend(loc="upper right")
plt.show()


def predict(X, w, b):
    y_pred = sigmoid(np.dot(X, w) + b)
    return y_pred >= 0.5


p = predict(X_mapped, w, b)
print('Accuracy of training model: %f' % (np.mean(p == y_train) * 100))
