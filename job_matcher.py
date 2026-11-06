import pandas as pd

from resume_parser import extract_text_from_pdf
from skill_extractor import extract_skills


def calculate_match_score(resume_skills, required_skills):

    matched_skills = []

    for skill in required_skills:
        if skill.lower() in [s.lower() for s in resume_skills]:
            matched_skills.append(skill)

    if len(required_skills) == 0:
        score = 0
    else:
        score = (len(matched_skills) / len(required_skills)) * 100

    return score, matched_skills


# Load job dataset
jobs = pd.read_csv("data/job_descriptions.csv")


# Extract resume text
resume_path = "data/resumes/resume.pdf"

resume_text = extract_text_from_pdf(resume_path)


# Extract resume skills
resume_skills = extract_skills(resume_text)


print("\n----- RESUME SKILLS -----")

for skill in resume_skills:
    print("✓", skill)


print("\n----- JOB RECOMMENDATIONS -----")


results = []


for _, job in jobs.iterrows():

    required_skills = [
        skill.strip()
        for skill in job["required_skills"].split(",")
    ]

    score, matched_skills = calculate_match_score(
        resume_skills,
        required_skills
    )

    results.append({
        "Job Title": job["job_title"],
        "Company": job["company"],
        "Match Score": score,
        "Matched Skills": ", ".join(matched_skills)
    })


# Convert results into DataFrame
results_df = pd.DataFrame(results)


# Sort jobs by highest match
results_df = results_df.sort_values(
    by="Match Score",
    ascending=False
)


print(results_df.to_string(index=False))