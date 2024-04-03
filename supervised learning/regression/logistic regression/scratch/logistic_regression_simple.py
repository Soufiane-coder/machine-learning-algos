import numpy as np
import random as rd
# Generate random X values between 0 and 1
X = np.random.rand(100)
# Generate Y values based on the linear function with added noise
Y = [rd.randint(0, 1) for t in X]

# X = np.array([[1], [2], [3], [4], [5], [5.1], [5.5], [6], [7], [8], [9]])
# Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# définition de la fonction sigmoid


def sigmoid(z):
    return 1 / (1 + np.exp(z))

# Implémentation de modèle
def model(X, Y, epochs):
    w0 = 0
    w1 = 0
    liste_mse = []
    lr = 0.01
    for i in range(epochs):
        linear_predicted = w0 + w1 * X
        y_predicted = sigmoid(linear_predicted)
        mse = (1 / len(X)) * sum(pow(Y - y_predicted, 2))
        liste_mse.append(mse)
        dw1 = (2 / len(X) * sum(-X * (Y - y_predicted)))
        dw0 = (2 / len(X) * sum(-(Y - y_predicted)))
        w0 = w0 - dw0 * lr
        w1 = w1 - dw1 * lr
    return liste_mse, w0, w1


def predict(X, w0, w1):
    linear_predicted = w0 + w1 * X
    y_pred = sigmoid(linear_predicted)
    print(y_pred)
    class_pred = [0 if (y <= 0.5) else 1 for y in y_pred]
    return class_pred


# Call the model function with X, Y, and 100 epochs
liste_mse, w0, w1 = model(X, Y, epochs=1000)
# Print the final values of w0 and w1
print("Final values: w0 = {}, w1 = {}".format(w0, w1))
class_pred = predict(X, w0, w1)
print(class_pred)
