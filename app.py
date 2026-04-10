from flask import Flask, request, jsonify
import re
import math
from collections import Counter
from datetime import datetime
import uuid

app = Flask(__name__)

SKILL_DATABASE = {
    "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust", "swift", "kotlin", "php", "scala", "r", "matlab"],
    "web_frontend": ["html", "css", "react", "angular", "vue", "svelte", "nextjs", "tailwind", "bootstrap", "jquery", "sass", "webpack"],
    "web_backend": ["flask", "django", "express", "fastapi", "spring", "nodejs", "rails", "laravel", "asp.net", "gin"],
    "databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb", "sqlite", "firebase"],
    "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "devops", "heroku"],
    "data_science": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "matplotlib", "seaborn", "spark", "hadoop"],
    "tools": ["git", "github", "jira", "confluence", "postman", "vs code", "linux", "agile", "scrum", "rest api"],
}

ALL_SKILLS = set()
for category_skills in SKILL_DATABASE.values():
    ALL_SKILLS.update(category_skills)

analysis_history = []


def extract_text_content(text):
    text = text.lower()
    text = re.sub(r'[^\w\s/#+.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text):
    text_lower = text.lower()
    found_skills = {}

    for category, skills in SKILL_DATABASE.items():
        for skill in skills:
            pattern = r'\b' + re.escape(skill.replace('+', r'\+')) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                found_skills[skill] = {
                    "category": category,
                    "count": len(matches),
                    "skill": skill,
                }
    return found_skills


def compute_tfidf_scores(text, corpus_size=100):
    words = text.lower().split()
    word_freq = Counter(words)
    total_words = len(words)

    scores = {}
    for word, count in word_freq.items():
        if len(word) < 2:
            continue
        tf = count / total_words
        idf = math.log(corpus_size / (1 + min(count, corpus_size // 2)))
        scores[word] = round(tf * idf, 4)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50])


def compute_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0, [], []

    resume_set = set(resume_skills.keys())
    jd_set = set(jd_skills.keys())

    matched = resume_set & jd_set
    missing = jd_set - resume_set
    extra = resume_set - jd_set

    weights = {"programming": 3, "web_backend": 2.5, "databases": 2, "cloud": 2.5, "data_science": 2, "web_frontend": 1.5, "tools": 1}

    weighted_matched = sum(weights.get(jd_skills[s]["category"], 1) for s in matched)
    weighted_total = sum(weights.get(jd_skills[s]["category"], 1) for s in jd_set)

    score = round((weighted_matched / weighted_total) * 100, 1) if weighted_total > 0 else 0

    missing_prioritized = sorted(
        [{"skill": s, "category": jd_skills[s]["category"], "priority": weights.get(jd_skills[s]["category"], 1)} for s in missing],
        key=lambda x: x["priority"],
        reverse=True,
    )

    return score, list(matched), missing_prioritized


def extract_sections(text):
    sections = {}
    section_headers = ["experience", "education", "skills", "projects", "summary", "certifications", "achievements", "objective"]

    lines = text.split('\n')
    current_section = "header"
    sections[current_section] = []

    for line in lines:
        line_clean = line.strip().lower()
        matched_header = None
        for header in section_headers:
            if header in line_clean and len(line_clean) < 30:
                matched_header = header
                break
        if matched_header:
            current_section = matched_header
            sections[current_section] = []
        else:
            if current_section in sections:
                sections[current_section].append(line.strip())
            else:
                sections[current_section] = [line.strip()]

    return {k: '\n'.join(v).strip() for k, v in sections.items() if v}


@app.route("/api/analyze/resume", methods=["POST"])
def analyze_resume():
    data = request.get_json()
    if not data or "resume_text" not in data:
        return jsonify({"error": "resume_text is required"}), 400

    text = data["resume_text"]
    processed = extract_text_content(text)

    skills = extract_skills(text)
    sections = extract_sections(text)
    tfidf = compute_tfidf_scores(processed)

    skills_by_category = {}
    for skill, info in skills.items():
        cat = info["category"]
        if cat not in skills_by_category:
            skills_by_category[cat] = []
        skills_by_category[cat].append(skill)

    word_count = len(processed.split())

    return jsonify({
        "word_count": word_count,
        "skills_found": len(skills),
        "skills": skills,
        "skills_by_category": skills_by_category,
        "sections_detected": list(sections.keys()),
        "keyword_scores": dict(list(tfidf.items())[:20]),
        "processed_at": datetime.utcnow().isoformat(),
    })


@app.route("/api/analyze/job", methods=["POST"])
def analyze_job():
    data = request.get_json()
    if not data or "job_description" not in data:
        return jsonify({"error": "job_description is required"}), 400

    text = data["job_description"]
    processed = extract_text_content(text)
    skills = extract_skills(text)
    tfidf = compute_tfidf_scores(processed)

    skills_by_category = {}
    for skill, info in skills.items():
        cat = info["category"]
        if cat not in skills_by_category:
            skills_by_category[cat] = []
        skills_by_category[cat].append(skill)

    return jsonify({
        "skills_required": len(skills),
        "skills": skills,
        "skills_by_category": skills_by_category,
        "keyword_scores": dict(list(tfidf.items())[:20]),
        "processed_at": datetime.utcnow().isoformat(),
    })


@app.route("/api/match", methods=["POST"])
def match_resume_to_job():
    data = request.get_json()
    if not data or "resume_text" not in data or "job_description" not in data:
        return jsonify({"error": "Both resume_text and job_description are required"}), 400

    resume_skills = extract_skills(data["resume_text"])
    jd_skills = extract_skills(data["job_description"])

    score, matched, missing = compute_match_score(resume_skills, jd_skills)

    result = {
        "match_id": str(uuid.uuid4())[:8],
        "match_score": score,
        "matched_skills": sorted(matched),
        "missing_skills": missing,
        "extra_skills": sorted(list(set(resume_skills.keys()) - set(jd_skills.keys()))),
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(jd_skills),
        "recommendation": (
            "Strong match" if score >= 75
            else "Good match with some gaps" if score >= 50
            else "Moderate match - significant upskilling needed" if score >= 25
            else "Low match - consider different role"
        ),
        "processed_at": datetime.utcnow().isoformat(),
    }

    analysis_history.append(result)
    return jsonify(result)


@app.route("/api/match/batch", methods=["POST"])
def batch_match():
    data = request.get_json()
    if not data or "pairs" not in data:
        return jsonify({"error": "pairs array is required"}), 400

    results = []
    for pair in data["pairs"]:
        resume_skills = extract_skills(pair.get("resume_text", ""))
        jd_skills = extract_skills(pair.get("job_description", ""))
        score, matched, missing = compute_match_score(resume_skills, jd_skills)
        results.append({
            "match_score": score,
            "matched_skills": sorted(matched),
            "missing_count": len(missing),
        })

    return jsonify({"total": len(results), "results": results})


@app.route("/api/skills/database", methods=["GET"])
def get_skill_database():
    return jsonify(SKILL_DATABASE)


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"total": len(analysis_history), "analyses": analysis_history[-20:]})


@app.route("/")
def dashboard():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5002)
