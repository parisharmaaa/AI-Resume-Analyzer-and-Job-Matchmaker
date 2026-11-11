from flask import Flask , render_template , request
from sentence_transformers import SentenceTransformer
import re
import fitz
import torch.nn.functional as F
app = Flask("resume analyze")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

SKILLS = [
    "python", "java", "c++", "sql", "mysql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "pandas", "numpy",
    "tensorflow", "pytorch", "scikit-learn", "flask",
    "django", "fastapi", "html", "css", "javascript",
    "react", "node.js", "git", "github", "docker",
    "aws", "azure", "power bi", "tableau", "nlp",
    "computer vision", "generative ai", "llm"
]



def extract_resume(file):
    with fitz.open(stream = file.read() , filetype = 'pdf') as pdf:
        return "\n".join(page.get_text() for page in pdf).strip()


def get_skills(text):
    text = text.lower()
    return {
        skill for skill in SKILLS
        if re.search(r'(?<!\w)' + re.escape(skill) + r'(?!\w)', text)

    }
    
def extract_section(text, section_names):
    lines = text.splitlines()

    headings = {
        "summary", "professional summary",
        "skills", "technical skills",
        "education", "projects",
        "experience", "work experience",
        "professional experience", "internships",
        "certifications"
    }

    for i, line in enumerate(lines):
        heading = line.strip().lower().rstrip(":")

        if heading in section_names:
            section = []

            for next_line in lines[i + 1:]:
                if next_line.strip().lower().rstrip(":") in headings:
                    break

                section.append(next_line)

            return "\n".join(section).strip()

    return ""

def analyze_education(resume_text, job_description):
    education = extract_section(
        resume_text, {"education"}
    ).lower()

    terms = [
        "computer science",
        "computer engineering",
        "information technology",
        "artificial intelligence",
        "b.tech",
        "bachelor",
        "b.e."
    ]

    required = [
        term for term in terms
        if term in job_description.lower()
    ]

    matched = [
        term for term in required
        if term in education
    ]

    return {
        "education_found": bool(education),
        "education_matches": matched
    }

def analyze_experience(resume_text):
    experience = extract_section(
        resume_text,
        {"experience", "work experience",
         "professional experience", "internships"}
    )

    return {
        "experience_found": bool(experience),
        "experience_status": (
            "Experience section detected"
            if experience
            else "No dedicated experience section detected"
        )
    }


def analyze_projects(resume_text, job_description):
    projects = extract_section(
        resume_text, {"projects"}
    )

    if not projects:
        return 0

    project_embedding = get_embedding(projects)
    job_embedding = get_embedding(job_description)

    similarity = (
        project_embedding * job_embedding
    ).sum().item()

    return round(max(0, min(1, similarity)) * 100)

def get_embedding(text):
    tokenizer = model.tokenizer
    tokens = tokenizer(
        text , add_special_tokens = False

    )["input_ids"]

    chunks = [
        tokenizer.decode(tokens[i:i + 200])
        for i in range(0, len(tokens), 200)

    ]

    embeddings = model.encode(chunks , convert_to_tensor = True)
    return F.normalize(embeddings.mean(dim=0), p=2 , dim = 0)


def analyze_resume(resume_text, job_description):
    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(job_description)

    similarity = (resume_embedding * job_embedding).sum().item()
    match_score = round(max(0, min(1, similarity)) * 100)

    resume_skills = get_skills(resume_text)
    job_skills = get_skills(job_description)

    matched_skills = sorted(resume_skills & job_skills)
    missing_skills = sorted(job_skills - resume_skills)

    suggestions = [
        f"Consider learning or developing your {skill} skills."
        for skill in missing_skills
    ]

    if not missing_skills:
        suggestions = ["You have all the required skills for this position!"]

    # Phase 4 analysis
    education_results = analyze_education(
        resume_text, job_description
    )

    experience_results = analyze_experience(resume_text)

    project_score = analyze_projects(
        resume_text, job_description
    )

    return {
        "match_score": match_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "education_found": education_results["education_found"],
        "education_matches": education_results["education_matches"],
        "experience_found": experience_results["experience_found"],
        "experience_status": experience_results["experience_status"],
        "project_score": project_score
    }

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method =="GET":
        return render_template("analyze.html")

    resume = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not resume or not resume.filename:
        return render_template("index.html", error="Please upload a resume file.")

    if not resume.filename.lower().endswith(".pdf"):
        return render_template("analyze.html", error = "only PDF files are allowed.")

    if len(job_description) < 30:
        return render_template("analyze.html", error= "enter detailed job description.")
        
    try:
        resume_text = extract_resume(resume)

        if not resume_text:
            return render_template(
                "analyze.html",
                error = "no readable text found. please upload a text-based PDF"
            )
        
        results = analyze_resume(resume_text, job_description)

        return render_template(
            "result.html",
            resume_name=resume.filename,
            resume_text=resume_text,
            job_description=job_description,
            **results
            )

    except Exception :
        app.logger.exception("resume processing failed")
        return render_template("analyze.html", error = "unable to process pdf.")

@app.errorhandler(413)
def file_too_large(error):
    return render_template("index.html", error="File size exceeds the limit of 5MB."), 413


if __name__ == "__main__":app.run(debug=True)