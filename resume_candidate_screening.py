# ==========================================================
# Resume Screening & Ranking System
# Part 1 - Data Loading, NLP, Skill Extraction & Similarity
# ==========================================================

# ==========================
# Import Required Libraries
# ==========================

import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Download NLTK Resources
# ==========================

nltk.download("punkt")
nltk.download("stopwords")

# ==========================
# Load Datasets
# ==========================

resume_df = pd.read_csv("resumes.csv")
job_df = pd.read_csv("job_descriptions.csv")

print("="*70)
print("Resume Dataset")
print("="*70)
print(resume_df.head())

print("\n")

print("="*70)
print("Job Description Dataset")
print("="*70)
print(job_df.head())

# ==========================
# Check Missing Values
# ==========================

print("\nChecking Missing Values...\n")

print(resume_df.isnull().sum())
print(job_df.isnull().sum())

resume_df.fillna("", inplace=True)
job_df.fillna("", inplace=True)

# ==========================
# Text Cleaning Function
# ==========================

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ==========================
# Clean Resume Text
# ==========================

resume_df["Clean_Resume"] = resume_df["Resume_Text"].apply(clean_text)

# ==========================
# Clean Job Description
# ==========================

job_df["Clean_JD"] = job_df["Job_Description"].apply(clean_text)

print("\nText Cleaning Completed Successfully.")

# ==========================
# Master Skill List
# ==========================

skills_list = [

"python",
"sql",
"excel",
"power bi",
"tableau",
"machine learning",
"deep learning",
"statistics",
"communication",
"pandas",
"numpy",
"scikit-learn",
"tensorflow",
"pytorch",
"django",
"flask",
"java",
"git",
"rest api",
"financial modeling",
"hr analytics",
"recruitment",
"dax",
"azure",
"etl",
"oop"

]

# ==========================
# Skill Extraction Function
# ==========================

def extract_skills(text):

    text = str(text).lower()

    found_skills = []

    for skill in skills_list:

        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text):

            found_skills.append(skill)

    return found_skills

# ==========================
# Extract Resume Skills
# ==========================

resume_df["Extracted_Skills"] = resume_df["Resume_Text"].apply(extract_skills)

# ==========================
# Extract JD Skills
# ==========================

job_df["JD_Skills"] = job_df["Job_Description"].apply(extract_skills)

print("\nSkill Extraction Completed.")

print("\nSample Resume Skills\n")

print(resume_df[["Name","Extracted_Skills"]].head())

# ==========================
# Save Cleaned Dataset
# ==========================

resume_df.to_csv("clean_resume.csv",index=False)

job_df.to_csv("clean_job.csv",index=False)

# ==========================================================
# Select Job Role
# ==========================================================

print("\n")

print("="*70)
print("Available Job Roles")
print("="*70)

print(job_df[["Job_ID","Job_Title"]])

job_id = int(input("\nEnter Job ID : "))

selected_job = job_df[job_df["Job_ID"]==job_id]

job_description = selected_job.iloc[0]["Clean_JD"]

required_skills = selected_job.iloc[0]["JD_Skills"]

print("\nSelected Job Role :")

print(selected_job.iloc[0]["Job_Title"])

print("\nRequired Skills :")

print(required_skills)

# ==========================================================
# TF-IDF Vectorization
# ==========================================================

documents = resume_df["Clean_Resume"].tolist()

documents.append(job_description)

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(documents)

resume_vectors = tfidf_matrix[:-1]

job_vector = tfidf_matrix[-1]

print("\nTF-IDF Vectorization Completed.")

# ==========================================================
# Cosine Similarity
# ==========================================================

similarity_scores = cosine_similarity(

resume_vectors,

job_vector

)

resume_df["Similarity_Score"] = (

similarity_scores.flatten()*100

).round(2)

print("\nSimilarity Scores Generated Successfully.\n")

print(

resume_df[
[
"Name",
"Similarity_Score"
]
].head()

)
# ==========================================================
# Resume Screening & Ranking System
# Part 2 - Scoring, Ranking & Visualization
# ==========================================================

# ==========================================================
# Matched Skills
# ==========================================================

def matched_skills(candidate_skills):
    return list(set(candidate_skills).intersection(set(required_skills)))

resume_df["Matched_Skills"] = resume_df["Extracted_Skills"].apply(matched_skills)

# ==========================================================
# Missing Skills
# ==========================================================

def missing_skills(candidate_skills):
    return list(set(required_skills) - set(candidate_skills))

resume_df["Missing_Skills"] = resume_df["Extracted_Skills"].apply(missing_skills)

# ==========================================================
# Skill Counts
# ==========================================================

resume_df["Matched_Count"] = resume_df["Matched_Skills"].apply(len)
resume_df["Missing_Count"] = resume_df["Missing_Skills"].apply(len)

# ==========================================================
# Skill Score
# ==========================================================

total_required = len(required_skills)

resume_df["Skill_Score"] = (
    resume_df["Matched_Count"] / total_required
) * 100

# ==========================================================
# Experience Score
# ==========================================================

def experience_score(exp):

    exp = str(exp).lower()

    if "fresher" in exp:
        return 40

    elif "1" in exp:
        return 60

    elif "2" in exp:
        return 75

    elif "3" in exp:
        return 90

    elif "4" in exp:
        return 100

    else:
        return 80

resume_df["Experience_Score"] = resume_df["Experience"].apply(experience_score)

# ==========================================================
# Final Score
# ==========================================================

resume_df["Final_Score"] = (

    0.60 * resume_df["Similarity_Score"]

    +

    0.30 * resume_df["Skill_Score"]

    +

    0.10 * resume_df["Experience_Score"]

).round(2)

# ==========================================================
# Recommendation
# ==========================================================

def recommendation(score):

    if score >= 85:
        return "Excellent Match"

    elif score >= 70:
        return "Good Match"

    elif score >= 50:
        return "Average Match"

    else:
        return "Low Match"

resume_df["Recommendation"] = resume_df["Final_Score"].apply(recommendation)

# ==========================================================
# Ranking
# ==========================================================

resume_df = resume_df.sort_values(
    by="Final_Score",
    ascending=False
)

resume_df.reset_index(drop=True, inplace=True)

resume_df["Rank"] = resume_df.index + 1

# ==========================================================
# Final Output
# ==========================================================

print("\n")
print("="*100)
print("FINAL CANDIDATE RANKING")
print("="*100)

print(

resume_df[
[
"Rank",
"Name",
"Similarity_Score",
"Skill_Score",
"Experience_Score",
"Final_Score",
"Recommendation"
]
]

)

# ==========================================================
# Candidate Skill Analysis
# ==========================================================

print("\n")
print("="*100)
print("MATCHED & MISSING SKILLS")
print("="*100)

print(

resume_df[
[
"Name",
"Matched_Skills",
"Missing_Skills"
]
]

)

# ==========================================================
# Save Results
# ==========================================================

resume_df.to_csv(
    "candidate_scores.csv",
    index=False
)

print("\ncandidate_scores.csv generated successfully.")

# ==========================================================
# Top 10 Candidates
# ==========================================================

print("\n")
print("="*100)
print("TOP 10 CANDIDATES")
print("="*100)

print(

resume_df[
[
"Rank",
"Name",
"Final_Score",
"Recommendation"
]
].head(10)

)

# ==========================================================
# Visualization
# ==========================================================

top10 = resume_df.head(10)

plt.figure(figsize=(12,6))

plt.bar(
    top10["Name"],
    top10["Final_Score"]
)

plt.xticks(rotation=45)

plt.xlabel("Candidates")

plt.ylabel("Final Score")

plt.title("Top 10 Ranked Candidates")

plt.tight_layout()

plt.show()

# ==========================================================
# Candidate Summary
# ==========================================================

print("\n")
print("="*100)
print("CANDIDATE SUMMARY")
print("="*100)

for index, row in resume_df.iterrows():

    print(f"\nRank : {row['Rank']}")
    print(f"Candidate : {row['Name']}")
    print(f"Similarity Score : {row['Similarity_Score']:.2f}%")
    print(f"Skill Score : {row['Skill_Score']:.2f}%")
    print(f"Experience Score : {row['Experience_Score']}")
    print(f"Final Score : {row['Final_Score']:.2f}%")
    print(f"Recommendation : {row['Recommendation']}")
    print(f"Matched Skills : {', '.join(row['Matched_Skills'])}")

    if len(row["Missing_Skills"]) == 0:
        print("Missing Skills : None")
    else:
        print(f"Missing Skills : {', '.join(row['Missing_Skills'])}")

    print("-"*100)

print("\n")
print("="*100)
print("RESUME SCREENING COMPLETED SUCCESSFULLY")
print("="*100)
