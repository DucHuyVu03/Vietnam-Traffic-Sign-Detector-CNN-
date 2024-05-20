from pathlib import Path
def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Remove newline characters from each line
        labels = [line.strip() for line in lines]
    return labels

data_folder = Path('VietnamTrafficSigns')
classes = data_folder/'classes.txt'
print(classes)
label = read_labels(classes)
print(label)