import numpy as np
import matplotlib.pyplot as plt

# Part a: Verify that 0 is a root of multiplicity 2 of f(x)
def f1(x):
    return np.exp(2 * np.sin(x)) - 2 * x - 1

def f1_prime(x):
    return 2 * np.exp(2 * np.sin(x)) * np.cos(x) - 2

def f1_double_prime(x):
    return 4 * np.exp(2 * np.sin(x)) * np.cos(x)**2 - 2 * np.exp(2 * np.sin(x)) * np.sin(x)

# Part a calculations
print(" Part a: Verifying multiplicity ")
x = 0
f1_0 = f1(x)  # f(0)
f1_prime_0 = f1_prime(x)  # f'(0)
f1_doubleprime_0 = f1_double_prime(x)  # f''(0)

print(f"f(0) = {f1_0}")
print(f"f'(0) = {f1_prime_0}")
print(f"f''(0) = {f1_doubleprime_0}")
if f1_0 == 0 and f1_prime_0 == 0 and f1_doubleprime_0 != 0:
    print("Verified: 0 is a root of multiplicity 2.")
else:
    print("Verification failed.")

# Part b: Apply Newton's Method and Modified Newton's Method for f1(x)
def newton_method(func, func_prime, x0, iterations):
    """Standard Newton's Method"""
    results = [x0]
    for _ in range(iterations):
        x0 -= func(x0) / func_prime(x0)
        results.append(x0)
    return results

def modified_newton_method(func, func_prime, x0, iterations):
    """Modified Newton's Method"""
    results = [x0]
    for _ in range(iterations):
        x0 -= 2 * func(x0) / func_prime(x0)
        results.append(x0)
    return results

print("\n Part b: Newton's Methods for f1(x) ")
x0_f1 = 0.1  # Initial guess
iterations = 9

# Standard Newton's Method
newton_f1 = newton_method(f1, f1_prime, x0_f1, iterations)

# Modified Newton's Method
modified_newton_f1 = modified_newton_method(f1, f1_prime, x0_f1, iterations)

print("Standard Newton's Method Results:", newton_f1)
print("Modified Newton's Method Results:", modified_newton_f1)

# Part c: Use Modified Newton's Method for f2(x)
def f2(x):
    return -8 * x**2 / (3 * x**2 + 1)

def f2_prime(x):
    # Derivative manually derived
    numerator = -16 * x * (3 * x**2 + 1) + 48 * x**3
    denominator = (3 * x**2 + 1)**2
    return numerator / denominator

# Updated Modified Newton's Method to handle division by zero
def modified_newton_method_safe(func, func_prime, x0, iterations):
    """Modified Newton's Method with division by zero handling."""
    results = [x0]
    for _ in range(iterations):
        derivative = func_prime(x0)
        if derivative == 0:  # Handle zero derivative
            print(f"Division by zero encountered at x = {x0}. Stopping iterations.")
            break
        x0 -= 2 * func(x0) / derivative
        results.append(x0)
    return results

print("\n Part c: Modified Newton's Method for f2(x) ")
x0_f2 = 0.15  # Initial guess

# Standard Newton's Method for f2(x)
newton_f2 = newton_method(f2, f2_prime, x0_f2, iterations)

# Modified Newton's Method for f2(x) with safe handling
modified_newton_f2 = modified_newton_method_safe(f2, f2_prime, x0_f2, iterations)

print("Standard Newton's Method Results for f2(x):", newton_f2)
print("Modified Newton's Method Results for f2(x):", modified_newton_f2)

# Final results for comparison
print("\n Final Comparison ")
print(f"Final x9 for f1(x) using Standard Newton's Method: {newton_f1[-1]}")
print(f"Final x9 for f1(x) using Modified Newton's Method: {modified_newton_f1[-1]}")
if modified_newton_f2:
    print(f"Final x9 for f2(x) using Modified Newton's Method: {modified_newton_f2[-1]}")
if newton_f2:
    print(f"Final x9 for f2(x) using Standard Newton's Method: {newton_f2[-1]}")

# Plotting results for f1(x)
plt.figure(figsize=(8, 6))
plt.plot(range(len(newton_f1)), newton_f1, marker='o', label="Standard Newton's Method for f1(x)")
plt.plot(range(len(modified_newton_f1)), modified_newton_f1, marker='x', label="Modified Newton's Method for f1(x)")
plt.xlabel('Iteration')
plt.ylabel('x Value')
plt.title('Convergence of Newton’s Methods for f1(x)')
plt.legend()
plt.grid()
plt.show()

# Plotting results for f2(x)
plt.figure(figsize=(8, 6))
plt.plot(range(len(newton_f2)), newton_f2, marker='o', label="Standard Newton's Method for f2(x)")
plt.plot(range(len(modified_newton_f2)), modified_newton_f2, marker='x', label="Modified Newton's Method for f2(x)")
plt.xlabel('Iteration')
plt.ylabel('x Value')
plt.title('Convergence of Newton’s Methods for f2(x)')
plt.legend()
plt.grid()
plt.show()
