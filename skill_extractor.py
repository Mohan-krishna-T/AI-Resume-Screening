# --------------------------------------------------
# SKILL EXTRACTOR
# --------------------------------------------------

SKILLS = [
    "Python",
    "SQL",
    "Excel",
    "Power BI",
    "Tableau",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "TensorFlow",
    "PyTorch",
    "Statistics",
    "Data Analysis",
    "Data Visualization",
    "Power Query",
    "DAX",
    "MySQL",
    "PostgreSQL",
    "Git",
    "GitHub",
    "Flask",
    "Java",
    "HTML",
    "CSS",
    "JavaScript"
]


# --------------------------------------------------
# EXTRACT SKILLS FROM RESUME
# --------------------------------------------------

def extract_skills(resume_text):

    found_skills = []

    resume_text_lower = resume_text.lower()

    for skill in SKILLS:

        if skill.lower() in resume_text_lower:

            found_skills.append(skill)

    return found_skills


# --------------------------------------------------
# LEARNING RECOMMENDATIONS
# --------------------------------------------------

def get_learning_recommendations(missing_skills):

    recommendations = {}

    learning_map = {

        "python":
            "Learn Python programming, Pandas, NumPy and data structures.",

        "sql":
            "Learn SQL queries, joins, subqueries, CTEs and window functions.",

        "excel":
            "Learn Advanced Excel, Pivot Tables, Power Query and dashboard development.",

        "power bi":
            "Learn Power BI, DAX, Power Query, data modeling and dashboard development.",

        "tableau":
            "Learn Tableau dashboards, calculated fields, filters and data visualization.",

        "pandas":
            "Learn Pandas for data cleaning, transformation and analysis.",

        "numpy":
            "Learn NumPy arrays, mathematical operations and numerical computing.",

        "matplotlib":
            "Learn Matplotlib for creating charts and data visualizations.",

        "seaborn":
            "Learn Seaborn for statistical data visualization.",

        "scikit-learn":
            "Learn Scikit-learn, preprocessing, model training and model evaluation.",

        "machine learning":
            "Learn supervised learning, unsupervised learning and model evaluation.",

        "deep learning":
            "Learn neural networks, CNNs, RNNs and deep learning fundamentals.",

        "nlp":
            "Learn Natural Language Processing, text preprocessing, embeddings and transformers.",

        "tensorflow":
            "Learn TensorFlow, neural networks and deep learning model development.",

        "pytorch":
            "Learn PyTorch, tensors, neural networks and deep learning.",

        "statistics":
            "Learn descriptive statistics, probability, correlation and hypothesis testing.",

        "data analysis":
            "Learn data cleaning, exploratory data analysis and statistical analysis.",

        "data visualization":
            "Learn visualization principles, dashboards and effective chart selection.",

        "power query":
            "Learn Power Query for data extraction, cleaning and transformation.",

        "dax":
            "Learn DAX measures, calculated columns, CALCULATE and filter context.",

        "mysql":
            "Learn MySQL queries, joins, indexes, views and database design.",

        "postgresql":
            "Learn PostgreSQL queries, joins, CTEs and database management.",

        "git":
            "Learn Git commands, branching, commits and version control.",

        "github":
            "Learn GitHub repositories, branches, pull requests and project collaboration.",

        "flask":
            "Learn Flask routing, APIs, templates and Python web application development.",

        "java":
            "Learn Java fundamentals, OOP, collections and application development.",

        "html":
            "Learn HTML structure, semantic elements and forms.",

        "css":
            "Learn CSS layouts, Flexbox, Grid and responsive design.",

        "javascript":
            "Learn JavaScript fundamentals, DOM manipulation and web application development."
    }


    for skill in missing_skills:

        key = skill.lower().strip()

        if key in learning_map:

            recommendations[skill] = learning_map[key]

        else:

            recommendations[skill] = (
                f"Learn the fundamentals and practical applications of {skill}."
            )


    return recommendations