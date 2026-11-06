import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from resume_parser import extract_text_from_pdf
from skill_extractor import extract_skills


# --------------------------------------------------
# 1. LOAD JOB DATA
# --------------------------------------------------

jobs = pd.read_csv("data/job_descriptions.csv")


# --------------------------------------------------
# 2. EXTRACT RESUME TEXT
# --------------------------------------------------

resume_path = "data/resumes/resume.pdf"

resume_text = extract_text_from_pdf(resume_path)


# --------------------------------------------------
# 3. EXTRACT RESUME SKILLS
# --------------------------------------------------

resume_skills = extract_skills(resume_text)


print("\n----- RESUME SKILLS -----")

for skill in resume_skills:
    print("✓", skill)


# --------------------------------------------------
# 4. CALCULATE SKILL MATCH
# --------------------------------------------------

def calculate_skill_match(resume_skills, required_skills):

    matched_skills = []

    resume_skills_lower = [
        skill.lower()
        for skill in resume_skills
    ]

    for skill in required_skills:

        if skill.lower() in resume_skills_lower:
            matched_skills.append(skill)

    if len(required_skills) == 0:
        score = 0
    else:
        score = (
            len(matched_skills)
            / len(required_skills)
        ) * 100

    return score, matched_skills


# --------------------------------------------------
# 5. TF-IDF MATCHING
# --------------------------------------------------

documents = [resume_text] + jobs["description"].tolist()

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    documents
)

tfidf_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)[0] * 100


# --------------------------------------------------
# 6. SEMANTIC AI MATCHING
# --------------------------------------------------

print("\nLoading semantic AI model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

job_descriptions = jobs[
    "description"
].tolist()

resume_embedding = model.encode(
    [resume_text]
)

job_embeddings = model.encode(
    job_descriptions
)

semantic_scores = cosine_similarity(
    resume_embedding,
    job_embeddings
)[0] * 100


# --------------------------------------------------
# 7. CALCULATE FINAL SCORE
# --------------------------------------------------

final_results = []


for index, job in jobs.iterrows():

    required_skills = [
        skill.strip()
        for skill in job["required_skills"].split(",")
    ]

    skill_score, matched_skills = calculate_skill_match(
        resume_skills,
        required_skills
    )

    tfidf_score = tfidf_scores[index]

    semantic_score = semantic_scores[index]

    final_score = (
        (skill_score * 0.40)
        + (tfidf_score * 0.20)
        + (semantic_score * 0.40)
    )

    missing_skills = [
        skill
        for skill in required_skills
        if skill not in matched_skills
    ]

    final_results.append({

        "Job Title": job["job_title"],

        "Company": job["company"],

        "Skill Match": skill_score,

        "TF-IDF Score": tfidf_score,

        "Semantic AI Score": semantic_score,

        "Final Score": final_score,

        "Matched Skills": ", ".join(
            matched_skills
        ),

        "Missing Skills": ", ".join(
            missing_skills
        )
    })


# --------------------------------------------------
# 8. CREATE RESULTS DATAFRAME
# --------------------------------------------------

results = pd.DataFrame(final_results)


# --------------------------------------------------
# 9. SORT BY FINAL SCORE
# --------------------------------------------------

results = results.sort_values(
    by="Final Score",
    ascending=False
)


# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================================")
print("       AI RESUME SCREENING RESULTS")
print("==============================================")

for _, row in results.iterrows():

    print(
        f"\n{row['Job Title']} - {row['Company']}"
    )

    print(
        f"Final Score: "
        f"{row['Final Score']:.2f}%"
    )

    print(
        f"Skill Match: "
        f"{row['Skill Match']:.2f}%"
    )

    print(
        f"TF-IDF: "
        f"{row['TF-IDF Score']:.2f}%"
    )

    print(
        f"Semantic AI: "
        f"{row['Semantic AI Score']:.2f}%"
    )

    print(
        f"Matched Skills: "
        f"{row['Matched Skills']}"
    )

    print(
        f"Missing Skills: "
        f"{row['Missing Skills']}"
    )


# --------------------------------------------------
# 11. BEST RECOMMENDATION
# --------------------------------------------------

best_job = results.iloc[0]

print("\n==============================================")
print("             BEST RECOMMENDATION")
print("==============================================")

print(
    f"Job Role: {best_job['Job Title']}"
)

print(
    f"Company: {best_job['Company']}"
)

print(
    f"Match Score: "
    f"{best_job['Final Score']:.2f}%"
)