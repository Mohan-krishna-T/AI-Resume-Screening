import sys
import os

# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# IMPORT LIBRARIES
# --------------------------------------------------

import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from src.resume_parser import extract_text_from_uploaded_pdf
from src.skill_extractor import (
    extract_skills,
    get_learning_recommendations
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

h1 {
    font-size: 2.5rem;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 10px;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🤖 AI Resume Screening & Job Recommendation System"
)

st.write(
    "Upload your resume and let our AI system analyze "
    "your skills and recommend suitable job roles."
)


# --------------------------------------------------
# LOAD JOB DATA
# --------------------------------------------------

try:

    jobs = pd.read_csv(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "job_descriptions.csv"
        )
    )

except FileNotFoundError:

    st.error(
        "job_descriptions.csv was not found. "
        "Please check the data folder."
    )

    st.stop()


# --------------------------------------------------
# LOAD SEMANTIC AI MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("Loading AI model..."):

    model = load_model()


# --------------------------------------------------
# RESUME UPLOAD
# --------------------------------------------------

st.subheader("📄 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose a PDF resume",
    type=["pdf"]
)


# --------------------------------------------------
# ANALYZE RESUME
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"Resume uploaded: {uploaded_file.name}"
    )

    analyze_button = st.button(
        "🔍 Analyze Resume"
    )

    if analyze_button:

        # --------------------------------------------------
        # EXTRACT RESUME TEXT
        # --------------------------------------------------

        with st.spinner("Reading resume..."):

            resume_text = extract_text_from_uploaded_pdf(
                uploaded_file
            )


        if not resume_text.strip():

            st.error(
                "Could not extract text from this PDF. "
                "Please upload a text-based PDF."
            )

            st.stop()


        # --------------------------------------------------
        # EXTRACT SKILLS
        # --------------------------------------------------

        resume_skills = extract_skills(
            resume_text
        )


        # --------------------------------------------------
        # DISPLAY DETECTED SKILLS
        # --------------------------------------------------

        st.subheader(
            "🛠️ Skills Detected"
        )

        if resume_skills:

            st.write(
                ", ".join(resume_skills)
            )

        else:

            st.warning(
                "No predefined skills were detected."
            )


        # --------------------------------------------------
        # TF-IDF MATCHING
        # --------------------------------------------------

        with st.spinner(
            "Calculating TF-IDF similarity..."
        ):

            documents = (
                [resume_text]
                + jobs["description"].fillna("").tolist()
            )

            tfidf_vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            tfidf_matrix = (
                tfidf_vectorizer.fit_transform(
                    documents
                )
            )

            tfidf_scores = (
                cosine_similarity(
                    tfidf_matrix[0:1],
                    tfidf_matrix[1:]
                )[0] * 100
            )


        # --------------------------------------------------
        # SEMANTIC AI MATCHING
        # --------------------------------------------------

        with st.spinner(
            "Analyzing resume with Semantic AI..."
        ):

            resume_embedding = model.encode(
                [resume_text]
            )

            job_embeddings = model.encode(
                jobs["description"]
                .fillna("")
                .tolist()
            )

            semantic_scores = (
                cosine_similarity(
                    resume_embedding,
                    job_embeddings
                )[0] * 100
            )


        # --------------------------------------------------
        # FINAL SCORING
        # --------------------------------------------------

        results = []


        for index, job in jobs.reset_index(
            drop=True
        ).iterrows():

            # Required skills
            required_skills = [
                skill.strip()
                for skill in str(
                    job["required_skills"]
                ).split(",")
                if skill.strip()
            ]


            # Resume skills in lowercase
            resume_skills_lower = [
                skill.lower()
                for skill in resume_skills
            ]


            # Find matched skills
            matched_skills = []

            for skill in required_skills:

                if skill.lower() in resume_skills_lower:

                    matched_skills.append(
                        skill
                    )


            # Skill score
            if required_skills:

                skill_score = (
                    len(matched_skills)
                    / len(required_skills)
                ) * 100

            else:

                skill_score = 0


            # Missing skills
            missing_skills = [
                skill
                for skill in required_skills
                if skill not in matched_skills
            ]


            # Final AI score
            final_score = (
                skill_score * 0.40
                + tfidf_scores[index] * 0.20
                + semantic_scores[index] * 0.40
            )


            results.append({

                "Job Title":
                    job["job_title"],

                "Company":
                    job["company"],

                "Skill Match":
                    skill_score,

                "TF-IDF":
                    tfidf_scores[index],

                "Semantic AI":
                    semantic_scores[index],

                "Final Score":
                    final_score,

                "Matched Skills":
                    ", ".join(
                        matched_skills
                    ),

                "Missing Skills":
                    ", ".join(
                        missing_skills
                    )
            })


        # --------------------------------------------------
        # CREATE RESULTS DATAFRAME
        # --------------------------------------------------

        results_df = pd.DataFrame(
            results
        )


        # --------------------------------------------------
        # SORT JOBS
        # --------------------------------------------------

        results_df = results_df.sort_values(
            by="Final Score",
            ascending=False
        ).reset_index(drop=True)


        # --------------------------------------------------
        # BEST JOB
        # --------------------------------------------------

        best_job = results_df.iloc[0]


        # --------------------------------------------------
        # BEST JOB RECOMMENDATION
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "🏆 Best Job Recommendation"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Recommended Role",
                best_job["Job Title"]
            )


        with col2:

            st.metric(
                "Company",
                best_job["Company"]
            )


        with col3:

            st.metric(
                "AI Match Score",
                f"{best_job['Final Score']:.2f}%"
            )


        # --------------------------------------------------
        # SCORE BREAKDOWN
        # --------------------------------------------------

        st.subheader(
            "📊 AI Score Breakdown"
        )


        score_col1, score_col2, score_col3 = st.columns(3)


        with score_col1:

            st.metric(
                "Skill Match",
                f"{best_job['Skill Match']:.2f}%"
            )


        with score_col2:

            st.metric(
                "TF-IDF Similarity",
                f"{best_job['TF-IDF']:.2f}%"
            )


        with score_col3:

            st.metric(
                "Semantic AI",
                f"{best_job['Semantic AI']:.2f}%"
            )


        # --------------------------------------------------
        # SKILL ANALYSIS
        # --------------------------------------------------

        st.subheader(
            "🧠 Skill Analysis"
        )


        skill_col1, skill_col2 = st.columns(2)


        with skill_col1:

            st.write(
                "### ✅ Matched Skills"
            )

            if best_job["Matched Skills"]:

                for skill in best_job[
                    "Matched Skills"
                ].split(","):

                    st.write(
                        f"✓ {skill.strip()}"
                    )

            else:

                st.write("None")


        with skill_col2:

            st.write(
                "### ❌ Missing Skills"
            )

            if best_job["Missing Skills"]:

                for skill in best_job[
                    "Missing Skills"
                ].split(","):

                    st.write(
                        f"• {skill.strip()}"
                    )

            else:

                st.success(
                    "You have all the required skills!"
                )


        # --------------------------------------------------
        # LEARNING RECOMMENDATIONS
        # --------------------------------------------------

        st.subheader(
            "📚 Recommended Learning"
        )


        missing_skills_text = (
            best_job["Missing Skills"]
        )


        if missing_skills_text:

            missing_skills_list = [
                skill.strip()
                for skill in missing_skills_text.split(",")
                if skill.strip()
            ]


            recommendations = (
                get_learning_recommendations(
                    missing_skills_list
                )
            )


            for skill, recommendation in (
                recommendations.items()
            ):

                st.write(
                    f"### 🎯 {skill}"
                )

                st.info(
                    recommendation
                )

        else:

            st.success(
                "🎉 No major skill gaps found "
                "for this recommended role!"
            )


        # --------------------------------------------------
        # JOB RECOMMENDATIONS
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "💼 Job Recommendations"
        )


        display_df = results_df[
            [
                "Job Title",
                "Company",
                "Skill Match",
                "TF-IDF",
                "Semantic AI",
                "Final Score"
            ]
        ].copy()


        display_df[
            [
                "Skill Match",
                "TF-IDF",
                "Semantic AI",
                "Final Score"
            ]
        ] = display_df[
            [
                "Skill Match",
                "TF-IDF",
                "Semantic AI",
                "Final Score"
            ]
        ].round(2)


        st.dataframe(
            display_df,
            use_container_width=True
        )


        # --------------------------------------------------
        # JOB SCORE CHART
        # --------------------------------------------------

        st.subheader(
            "📈 Job Match Comparison"
        )


        chart_data = results_df[
            [
                "Job Title",
                "Final Score"
            ]
        ].set_index(
            "Job Title"
        )


        st.bar_chart(
            chart_data
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Resume Screening System | "
    "NLP + Machine Learning + Semantic AI"
)