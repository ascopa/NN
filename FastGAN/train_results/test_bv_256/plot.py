import matplotlib.pyplot as plt
import os
import numpy

fid = os.path.join(os.getcwd(), 'train_results', "test_bv_256", "fid.txt")    


# Read FID values from the file
with open(fid, 'r') as fid_file:
    fid_values = [float(line.strip()) for line in fid_file.readlines()]

# Define the interval for the x-axis
iteration_interval = 1000  # Change this to your preferred interval

# Create a list of iterations
iterations = list(range(1, len(fid_values) * iteration_interval + 1, iteration_interval))

# Plot the FID values by epoch
plt.plot(iterations, fid_values, marker='.', linestyle='-', color="red")
plt.xlabel('Ciclo de entrenamiento')
plt.ylabel('FID')
#plt.title('FID seg by Epoch')
plt.grid(True)

# Display the plot or save it to a file
plt.savefig("fid.png", dpi=600)
plt.show()

