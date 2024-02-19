import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load your image
image_path = "das_image_0.png"  # Replace with your actual image path

# Create a figure and axis
fig, ax = plt.subplots()

# Load the image (optional if you want to show the image in the plot)
img = plt.imread(image_path)
ax.imshow(img)

# Set axis labels
ax.set_xlabel("Muestras temporales")
ax.set_ylabel("Sensores")

# Set other plot configurations if needed
# For example, adjust the number of ticks on each axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))

ax.invert_yaxis()

# Save the plot as a PNG file with 400 DPI
output_path = "sinograma.png"
plt.savefig(output_path, dpi=400, bbox_inches='tight')

# Show the plot (optional)
plt.show()
