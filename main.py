import pandas as pd

# Load job description dataset
jobs = pd.read_csv("data/job_descriptions.csv")

# Display the dataset
print(jobs)

# Display job titles
print("\nAvailable Job Roles:")
print(jobs["job_title"])