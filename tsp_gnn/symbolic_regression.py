import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Step 1: Load the data from the CSV file
data = pd.read_csv('tsp_analysis_results.csv')

# Extract relevant columns: node_count (x), sample_size (y), and time (z)
X = data[['x (node_count)', 'y (sample_size)']].values  # Features (x, y)
y = data['z (time)'].values  # Target variable (z)

# x1 = node_count
# x2 = sample_size

# Step 2: Set up the symbolic regressor
model = PySRRegressor(
    niterations=100,  # Number of iterations to run the symbolic regression
    populations=100,  # Population size
    unary_operators=['square', "exp", "sqrt", 'cube', 'log', 'sin'], 
    binary_operators=["+", "*", "-", "/", "^"],
    verbosity=1,  # Increase verbosity to see intermediate outputs
)

# Step 3: Fit the model
model.fit(X, y)

# Step 4: Print the best equation found by symbolic regression
best_equation = model  # Best equation found by PySR
print("Best symbolic regression equation found:")
print(best_equation)

print(model.sympy())

# Step 5: Make predictions (optional, if you want to predict new values)
predictions = model.predict(X)

# Step 6: Calculate MSE, RMSE, MAE
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, predictions)

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

# Step 7: Visualize the prediction vs actual time
plt.figure(figsize=(8, 6))
plt.scatter(y, predictions, label="Predictions", alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label="Ideal line (y = x)")
plt.xlabel('Actual Time (z)')
plt.ylabel('Predicted Time (z)')
plt.title('Symbolic Regression: Actual vs Predicted Time')
plt.legend()

# Save the plot to a PNG file
plt.savefig("symbolic_regression_plot.png")

# Display the plot
plt.show()

# Optionally, save the best equation and metrics to a text file
with open('regression_results.txt', 'w') as f:
    f.write(f"Best Symbolic Regression Equation: {best_equation}\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
