import numpy as np
import matplotlib.pyplot as plt
# Generate random X values between 0 and 1
np.random.seed(42)

# Generate random X values between 0 and 10
X = 10 * np.random.rand(100, 1)

# Generate Y values based on a linear function with some random noise
# y = 3X + 5 + noise
noise = np.random.randn(100, 1)
Y = 3 * X + 5 + noise

# training dataset and labels
# train_input = np.array(data.x[0:500]).reshape(500, 1)
# train_output = np.array(data.y[0:500]).reshape(500, 1)

# # valid dataset and labels
# test_input = np.array(data.x[500:700]).reshape(199, 1)
# test_output = np.array(data.y[500:700]).reshape(199, 1)


def model(X, Y, epochs):
    w0 = 0
    w1 = 0
    liste_mse = []
    lr = 0.01
    for i in range(epochs):
        # Calculate predicted values using the linear regression equation
        y_predicted = w1*X + w0
        # Calculate Mean Squared Error (MSE)
        mse = (1 / len(X)) * sum((Y - y_predicted) ** 2)
        liste_mse.append(mse)
        # Partial derivative of MSE w.r.t w1
        dw1 = (-2 / len(X)) * sum(X * (Y - y_predicted))
        # Partial derivative of MSE w.r.t w0
        dw0 = (-2 / len(X)) * sum(Y - y_predicted)
        w0 = w0 - lr * dw0  # Update w0 using gradient descent
        w1 = w1 - lr * dw1  # Update w1 using gradient descent
    return liste_mse, w0, w1


# Call the model function with X, Y, and 1000 epochs
liste_mse, w0, w1 = model(X, Y, epochs=1000)

# Print the final values of w0 and w1
print("Final values: w0 = {}, w1 = {}".format(w0, w1))

line_x = X  # Generate x values for the line
line_y = w1 * line_x + w0  # Calculate y values using the line equation

# Plot the original data points
plt.scatter(X, Y, color='green', label='Data')

# Plot the line
plt.plot(line_x, line_y, color='red', label='Fitted Line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()

# Show plot
plt.show()
