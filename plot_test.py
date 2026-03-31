import matplotlib.pyplot as plt
import numpy as np

# Define x values
x = np.linspace(-10, 10, 400)
y = x**2

# Plot
plt.plot(x, y)
plt.title('y = x^2')
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()
