import csv

input_path = "data/gesture_dataset.csv"
output_path = "data/gesture_dataset_fixed.csv"

with open(input_path, "r", newline="") as file:
    rows = list(csv.reader(file))

# The current first row is actually the first data sample
first_row = rows[0]

# Create the correct header
header = [f"feature_{i}" for i in range(63)]
header.append("label")

# Put the correct header before ALL existing rows
rows = [header] + rows

with open(output_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print(f"Fixed dataset saved to: {output_path}")
print(f"Total samples: {len(rows) - 1}")
print(f"Total columns: {len(rows[0])}")