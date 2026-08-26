import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config_path = os.path.join(project_root, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

raw_data_path = os.path.join(project_root, config["Data"]["raw_data_path"])

clean_data_path = os.path.join(project_root,config["Data"]["clean_data_path"])

visualization_dir = os.path.join(project_root, config["Data"]["visualization_dir"])
os.makedirs(visualization_dir,exist_ok=True)

clustering_path = os.path.join(project_root, config["Data"]["clustering_charts"])
os.makedirs(clustering_path,exist_ok = True)


if __name__ == '__main__':
    print("Raw Data Path:", raw_data_path)
    print("Clean Data Path:", clean_data_path)
    print("Visualization Path:", visualization_dir)
    print("Clustering Path", clustering_path)