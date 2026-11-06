import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_text_from_pdf


# Load job dataset
jobs = pd.read_csv("data/job_descriptions.csv")


# Extract resume text
resume_path = "data/resumes/resume.pdf"

resume_text = extract_text_from_pdf(resume_path)


# Load AI embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create text list
job_descriptions = jobs["description"].tolist()


# Convert resume into an embedding
resume_embedding = model.encode([resume_text])


# Convert job descriptions into embeddings
job_embeddings = model.encode(job_descriptions)


# Calculate semantic similarity
similarity_scores = cosine_similarity(
    resume_embedding,
    job_embeddings
)[0]


# Convert similarity into percentage
jobs["AI Match Score"] = similarity_scores * 100


# Sort by highest score
jobs = jobs.sort_values(
    by="AI Match Score",
    ascending=False
)


# Display recommendations
print("\n----- SEMANTIC AI JOB RECOMMENDATIONS -----")

for _, job in jobs.iterrows():

    print(
        f"{job['job_title']} - "
        f"{job['company']} - "
        f"{job['AI Match Score']:.2f}%"
    )