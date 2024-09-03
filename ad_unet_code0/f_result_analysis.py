import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# Read the CSV data into a DataFrame
df = pd.read_csv("save_model_1/data.csv")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['accuracy'], marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='r', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.grid(True)
plt.show()