import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Gradient Descent function
def gradient_descent(init_guess, learning_rate, cost_function):
    p = init_guess
    iteration = 0
    p_values = []
    iterations = []

    while True:
        gradient = sp.diff(cost_function, x).subs(x, p)
        p = p - learning_rate * gradient
        iteration += 1
        p_values.append(p)
        iterations.append(iteration)
        print("Iteration ", iteration, ": p = ", p)
        if iteration == 50:
            break

    return p, iteration, p_values, iterations

# Plotting function
def plot_gradient_descent(cost_function, p_values, iterations):
    plt.figure(figsize=(12, 6))

    # Plot p values vs iterations
    plt.subplot(1, 2, 1)
    cost_values = [cost_function.subs(x, p_val) for p_val in p_values]
    plt.plot(p_values, cost_values, marker='o')
    plt.title('Gradient Descent Path')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot gradient descent path
    plt.subplot(1, 2, 2)
    plt.plot(p_values, marker='o')
    plt.title('p values vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('p values')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    init_guess = 0.5
    learning_rate = 0.01
    x = sp.Symbol('x')
    cost_function = 3.5*x**2 - 14*x + 14  # Example cost function

    p, iteration, p_values, iterations = gradient_descent(init_guess, learning_rate, cost_function)
    plot_gradient_descent(cost_function, p_values, iterations)
