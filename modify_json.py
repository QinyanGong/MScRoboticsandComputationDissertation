import json
from pathlib import Path

# Define the path to the transforms.json file
json_path = Path("/path/to/your/transforms.json")

# Load the JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# Append '.png' to each file_path in the frames
for frame in data["frames"]:
    frame["file_path"] += ".png"

# Save the modified JSON back to the file
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print("Updated file_path entries in transforms.json")
