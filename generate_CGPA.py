import random

import numpy as np
import pandas as pd

num_of_people = 500
with open("names.txt", "r") as names_file:
    names = names_file.readlines()
sample_names = random.sample(names, num_of_people)

mean_cgpa = 2.6
std_dev_cgpa = 0.7
cgpa_data = np.random.normal(mean_cgpa, std_dev_cgpa, num_of_people)
cgpa_data = np.clip(cgpa_data, 1.0, 4.0)

people_data = [{"Name": sample_names[i].strip().capitalize(), "Past CGPA 1": cgpa} for i, cgpa in enumerate(cgpa_data)]
df = pd.DataFrame(people_data)
df["Past CGPA 2"] = df["Past CGPA 1"].apply(lambda x: random.uniform(0.8, 1.2) * x)
df["Current CGPA"] = df["Past CGPA 2"].apply(lambda x: random.uniform(0.9, 1.1) * x)


df["Past CGPA 1"] = df["Past CGPA 1"].round(2)
df["Past CGPA 2"] = df["Past CGPA 2"].round(2)
df["Current CGPA"] = df["Current CGPA"].round(2)

csv_filename = "cgpa.csv"
df.to_csv(csv_filename, index=False)

print(f"Dataset saved to {csv_filename}")
