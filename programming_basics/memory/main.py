# Simulating the dynamic allocation of an array
size = 5
arr = [None] * size  # Create a list with 5 None elements (uninitialized-like behavior)

# Set values in the "allocated" memory
for i in range(size):
    arr[i] = i * 2  # Assign values dynamically

# Print the values
for i in range(size):
    print(f"arr[{i}] = {arr[i]}")

# No need to free memory explicitly; Python's garbage collector handles it automatically
