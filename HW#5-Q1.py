import numpy as np
import matplotlib.pyplot as plt

# C
N = 100  # N = 100 or Total population number
q = 0.95  # q = 0.95 or Probability of an individual receiving a negative test result.

# Function for the avg num of tests
def average_tests(x):
    return N * (1 - q**x + 1 / x)

# Range of x values
x = np.linspace(1, 150, 1000)
y = [average_tests(x) for x in x]

# Determine the value of x that minimizes the average number of tests required.
o_x = x[np.argmin(y)]
min_tests = min(y)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Avg Num of Tests")
plt.axvline(o_x, color='green', linestyle='--', label=f"Optimal x ≈ {o_x:.2f}")
plt.scatter(o_x, min_tests, color='yellow', edgecolor='black', label=f"Min Tests ≈ {min_tests:.2f}")
plt.title("Optimization of Group Size x for Minimal Avg Num of Tests")
plt.xlabel("Group Size (x)")
plt.ylabel("Avg Num of Tests")
plt.legend()
plt.grid()
plt.show()

o_x, min_tests
