import os
import shutil
import time
from datetime import datetime
from a import function_a
from b import function_b
from c import function_c

# Set your threshold values
threshold1 = 10
threshold2 = 20

# Function to execute the functions and save outputs
def execute_and_save():
    # Execute function_a
    output_a = function_a()

    for _ in range(100):
        # Execute functions from b.py and c.py
        output_b = function_b()
        output_c = function_c()

        # Create a timestamp for the folder name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        folder_name = timestamp[:12]

        # Create a new folder
        os.makedirs(folder_name)

        # Save outputs to files in the new folder
        with open(os.path.join(folder_name, 'aaaa.h5ad'), 'w') as file_a:
            file_a.write(str(output_a))

        with open(os.path.join(folder_name, 'bbbb.pth'), 'w') as file_b:
            file_b.write(str(output_b))

        # Check conditions
        if output_a > threshold1 and output_b > threshold2:
            print("Conditions satisfied. Task completed.")
            return True
        else:
            print("Conditions not satisfied. Retrying...")

    return False

# Execute functions until conditions are satisfied
while not execute_and_save():
    # Continue to the next iteration of the loop
    pass

# Continue with the rest of your program if needed
