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
    match_score = round(max(0,min(1, similarity))*100)

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

    return {
        "match_score": match_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions
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


if __name__ == "__main__":
    app.run(debug = True)
