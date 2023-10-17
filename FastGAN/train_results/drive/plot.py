import matplotlib.pyplot as plt

# Read FID values from the file
with open('fid.txt', 'r') as fid_file:
    fid_values = [float(line.strip()) for line in fid_file.readlines()]

# Create a list of epochs (assuming one FID value per epoch)
epochs = list(range(1, len(fid_values) + 1))

# Plot the FID values by epoch
plt.plot(epochs, fid_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('FID Value')
plt.title('FID Values by Epoch')
plt.grid(True)

# Display the plot or save it to a file
plt.show()
