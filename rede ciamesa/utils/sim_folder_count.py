import os
import re

def find_highest_simulation_number(directory):
    highest_number = 0
    simulation_pattern = re.compile(r'^simulation_(\d+)$')

    try:
        # List all items in the given directory
        for item in os.listdir(directory):
            # Check if the item is a directory and matches the pattern
            if os.path.isdir(os.path.join(directory, item)):
                match = simulation_pattern.match(item)
                if match:
                    # Extract the number from the directory name
                    number = int(match.group(1))
                    # Update the highest number if this one is higher
                    if number > highest_number:
                        highest_number = number
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")

    return highest_number

if __name__ == "__main__":
    # Example usage:
    directory_path = './'
    highest_simulation_number = find_highest_simulation_number(directory_path)
    print(f"The highest simulation number is: {highest_simulation_number}")