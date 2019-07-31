import numpy as np

# gradient descent for 1-c
def compute_gradient_descent(x, y):
    # define w and b as all zeors initially
    w = np.zeros((x.shape[1], 1))
    b = 0
    
    # define learning rate, alpha, and delta, and total number of iterations, 
    # total_iterations
    alpha = 0.00001
    delta = 1.0
    total_iterations = 1500
    
    # loop over 1500 times
    y_estimate = np.dot(x, w) + b
    residual = y_estimate - y
    
    for i in range(0, total_iterations):
        huber_loss_cost_value = np.where(residual <= delta, residual, (delta * (np.abs(residual)/residual)))
        w = w - (alpha / x.shape[0]) * np.dot(x.T, huber_loss_cost_value)
        b = b - (alpha / x.shape[0]) * ((huber_loss_cost_value).mean())
    
    return w, b
    
if __name__ == "__main__":
    w, b = compute_gradient_descent(x, y)