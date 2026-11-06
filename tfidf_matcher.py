import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_text_from_pdf


# Load job dataset
jobs = pd.read_csv("data/job_descriptions.csv")


# Extract resume text
resume_path = "data/resumes/resume.pdf"

resume_text = extract_text_from_pdf(resume_path)


# Create a list containing resume + job descriptions
documents = [resume_text] + jobs["description"].tolist()


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english"
)


# Convert text into numerical vectors
tfidf_matrix = vectorizer.fit_transform(documents)


# Compare resume with all job descriptions
similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)[0]


# Add similarity scores to dataset
jobs["Match Score"] = similarity_scores * 100


# Sort jobs by highest match
jobs = jobs.sort_values(
    by="Match Score",
    ascending=False
)


# Display results
print("\n----- AI JOB RECOMMENDATIONS -----")

for _, job in jobs.iterrows():

    print(
        f"{job['job_title']} - "
        f"{job['company']} - "
        f"{job['Match Score']:.2f}%"
    )