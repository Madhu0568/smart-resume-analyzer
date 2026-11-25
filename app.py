from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "analysis_history.db"
MAX_TEXT_LENGTH = 100_000
MAX_BATCH_SIZE = 50

app = Flask(__name__, static_folder="static")
app.config["JSON_SORT_KEYS"] = False

# Canonical skill name -> aliases users may write in resumes/job descriptions.
SKILL_DATABASE = {
    "programming": {
        "python": ["python", "python3"],
        "java": ["java"],
        "javascript": ["javascript", "js"],
        "typescript": ["typescript", "ts"],
        "c": ["c language", "programming in c"],
        "c++": ["c++", "cpp"],
        "c#": ["c#", "c sharp"],
        "ruby": ["ruby"],
        "go": ["golang", "go language"],
        "rust": ["rust"],
        "r": ["r programming", "r language"],
        "matlab": ["matlab"],
        "php": ["php"],
        "kotlin": ["kotlin"],
        "swift": ["swift"],
    },
    "web_frontend": {
        "html": ["html", "html5"],
        "css": ["css", "css3"],
        "react": ["react", "react.js", "reactjs"],
        "angular": ["angular"],
        "vue": ["vue", "vue.js", "vuejs"],
        "next.js": ["next.js", "nextjs"],
        "tailwind css": ["tailwind css", "tailwind"],
        "bootstrap": ["bootstrap"],
        "javascript": ["javascript", "js"],
    },
    "web_backend": {
        "flask": ["flask"],
        "django": ["django"],
        "fastapi": ["fastapi"],
        "express.js": ["express", "express.js", "expressjs"],
        "node.js": ["node.js", "nodejs", "node"],
        "spring": ["spring", "spring boot"],
        "rest api": ["rest api", "rest apis", "restful api", "restful apis"],
    },
    "databases": {
        "sql": ["sql"],
        "mysql": ["mysql"],
        "postgresql": ["postgresql", "postgres", "psql"],
        "sqlite": ["sqlite", "sqlite3"],
        "mongodb": ["mongodb", "mongo db"],
        "redis": ["redis"],
        "elasticsearch": ["elasticsearch"],
        "oracle": ["oracle database", "oracle sql"],
        "firebase": ["firebase"],
    },
    "cloud_devops": {
        "aws": ["aws", "amazon web services"],
        "azure": ["azure", "microsoft azure"],
        "gcp": ["gcp", "google cloud platform"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "terraform": ["terraform"],
        "jenkins": ["jenkins"],
        "ci/cd": ["ci/cd", "continuous integration", "continuous delivery"],
        "linux": ["linux"],
        "git": ["git"],
        "github": ["github"],
    },
    "data_science_ml": {
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch", "torch"],
        "keras": ["keras"],
        "matplotlib": ["matplotlib"],
        "seaborn": ["seaborn"],
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning"],
        "natural language processing": ["natural language processing", "nlp"],
        "tf-idf": ["tf-idf", "tfidf", "term frequency inverse document frequency"],
        "excel": ["excel", "microsoft excel"],
        "power bi": ["power bi", "powerbi"],
        "tableau": ["tableau"],
        "statistics": ["statistics", "statistical analysis"],
    },
    "tools": {
        "postman": ["postman"],
        "jira": ["jira"],
        "confluence": ["confluence"],
        "vs code": ["vs code", "visual studio code"],
        "agile": ["agile"],
        "scrum": ["scrum"],
        "spark": ["apache spark", "spark"],
        "hadoop": ["hadoop"],
    },
}

CATEGORY_WEIGHTS = {
    "programming": 3.0,
    "web_backend": 2.5,
    "cloud_devops": 2.5,
    "databases": 2.0,
    "data_science_ml": 2.0,
    "web_frontend": 1.5,
    "llm_ai_engineering": 3.0,
    "testing": 1.5,
    "tools": 1.0,
}

SECTION_ALIASES = {
    "summary": {"summary", "professional summary", "profile", "career objective", "objective"},
    "experience": {"experience", "work experience", "professional experience", "internship", "internships"},
    "education": {"education", "academic background", "academic details"},
    "skills": {"skills", "technical skills", "key skills", "core competencies"},
    "projects": {"projects", "academic projects", "personal projects"},
    "certifications": {"certifications", "certificates", "licenses and certifications"},
    "achievements": {"achievements", "awards", "accomplishments"},
}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def database_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def initialize_database() -> None:
    with database_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL UNIQUE,
                match_score REAL NOT NULL,
                recommendation TEXT NOT NULL,
                matched_skills TEXT NOT NULL,
                missing_skills TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def validate_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    value = value.strip()
    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} must be {MAX_TEXT_LENGTH} characters or fewer.")
    return value
def _compile_alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias.strip())

    if alias.strip().lower() == "c":
        return re.compile(
            r"(?<![A-Za-z0-9+#])c(?![A-Za-z0-9+#])",
            re.IGNORECASE
        )

    return re.compile(
        r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])",
        re.IGNORECASE
    )
COMPILED_SKILLS = []
for category, skill_map in SKILL_DATABASE.items():
    for canonical, aliases in skill_map.items():
        for alias in sorted(set([canonical] + aliases), key=len, reverse=True):
            COMPILED_SKILLS.append(
                (category, canonical, alias, _compile_alias_pattern(alias))
            )
# Prefer longer aliases first to avoid a shorter alias masking a longer phrase.
COMPILED_SKILLS.sort(key=lambda item: len(item[2]), reverse=True)

def extract_skills(text: str) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for category, canonical, alias, pattern in COMPILED_SKILLS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        if canonical not in found:
            found[canonical] = {
                "skill": canonical,
                "category": category,
                "count": 0,
            }
        found[canonical]["count"] += len(matches)
    return found

def normalize_heading(line: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z &]", "", line).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip(" :")

def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    alias_to_section = {
        alias: section
        for section, aliases in SECTION_ALIASES.items()
        for alias in aliases
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = normalize_heading(line)
        matched_section = alias_to_section.get(heading)
        if matched_section:
            current = matched_section
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if lines
    }

def extract_tfidf_keywords(text: str, top_n: int = 20) -> dict[str, float]:
    # Treat sentences/meaningful lines as documents so IDF is calculated
    # from document frequency within this text, not from a fabricated corpus size.
    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(chunk.strip().split()) >= 2
    ]
    if not chunks:
        chunks = [text.strip()]

    if not any(re.search(r"[A-Za-z]", chunk) for chunk in chunks):
        return {}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]*\b",
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        return {}

    terms = vectorizer.get_feature_names_out()
    scores = matrix.max(axis=0).toarray().ravel()
    ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
    return {term: round(float(score), 4) for term, score in ranked[:top_n]}

def group_skills_by_category(skills: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for skill, details in skills.items():
        grouped.setdefault(details["category"], []).append(skill)
    return {category: sorted(values) for category, values in grouped.items()}

def calculate_match(resume_text: str, job_description: str) -> dict[str, Any]:
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    resume_set = set(resume_skills)
    job_set = set(job_skills)
    matched = sorted(resume_set & job_set)
    missing = sorted(job_set - resume_set)
    extra = sorted(resume_set - job_set)

    total_weight = sum(
        CATEGORY_WEIGHTS.get(job_skills[skill]["category"], 1.0)
        for skill in job_set
    )
    matched_weight = sum(
        CATEGORY_WEIGHTS.get(job_skills[skill]["category"], 1.0)
        for skill in matched
    )
    score = round((matched_weight / total_weight) * 100, 1) if total_weight else 0.0

    missing_details = [
        {
            "skill": skill,
            "category": job_skills[skill]["category"],
            "priority": CATEGORY_WEIGHTS.get(job_skills[skill]["category"], 1.0),
        }
        for skill in missing
    ]
    missing_details.sort(key=lambda item: (-item["priority"], item["skill"]))

    if not job_set:
        recommendation = "No recognized skills found in the job description"
    elif score >= 75:
        recommendation = "Strong match"
    elif score >= 50:
        recommendation = "Good match with some gaps"
    elif score >= 25:
        recommendation = "Moderate match - significant upskilling needed"
    else:
        recommendation = "Low match - consider building the required skills"

    # TF-IDF is calculated over sentence-level chunks in each text.
    resume_keywords = extract_tfidf_keywords(resume_text, top_n=15)
    job_keywords = extract_tfidf_keywords(job_description, top_n=15)

    return {
        "match_id": str(uuid.uuid4())[:8],
        "match_score": score,
        "matched_skills": matched,
        "missing_skills": missing_details,
        "extra_skills": extra,
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills),
        "resume_keywords": resume_keywords,
        "job_keywords": job_keywords,
        "recommendation": recommendation,
        "processed_at": utc_now(),
    }

def save_history(result: dict[str, Any]) -> None:
    import json
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO analysis_history
            (match_id, match_score, recommendation, matched_skills, missing_skills, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result["match_id"],
                result["match_score"],
                result["recommendation"],
                json.dumps(result["matched_skills"]),
                json.dumps(result["missing_skills"]),
                result["processed_at"],
            ),
        )

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "details": str(error)}), 400

@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "Internal server error"}), 500

@app.get("/")
def dashboard():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Smart Resume Analyzer"})

@app.post("/api/analyze/resume")
def analyze_resume():
    data = request.get_json(silent=True) or {}
    try:
        text = validate_text(data.get("resume_text"), "resume_text")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    skills = extract_skills(text)
    sections = extract_sections(text)
    keywords = extract_tfidf_keywords(text)

    return jsonify({
        "word_count": len(text.split()),
        "skills_found": len(skills),
        "skills": skills,
        "skills_by_category": group_skills_by_category(skills),
        "sections_detected": list(sections.keys()),
        "sections": sections,
        "keyword_scores": keywords,
        "processed_at": utc_now(),
    })

@app.post("/api/analyze/job")
def analyze_job():
    data = request.get_json(silent=True) or {}
    try:
        text = validate_text(data.get("job_description"), "job_description")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    skills = extract_skills(text)
    return jsonify({
        "skills_required": len(skills),
        "skills": skills,
        "skills_by_category": group_skills_by_category(skills),
        "keyword_scores": extract_tfidf_keywords(text),
        "processed_at": utc_now(),
    })

@app.post("/api/match")
def match_resume_to_job():
    data = request.get_json(silent=True) or {}
    try:
        resume_text = validate_text(data.get("resume_text"), "resume_text")
        job_description = validate_text(data.get("job_description"), "job_description")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    result = calculate_match(resume_text, job_description)
    save_history(result)
    return jsonify(result)

@app.post("/api/match/batch")
def batch_match():
    data = request.get_json(silent=True) or {}
    pairs = data.get("pairs")

    if not isinstance(pairs, list) or not pairs:
        return jsonify({"error": "pairs must be a non-empty array"}), 400
    if len(pairs) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Maximum batch size is {MAX_BATCH_SIZE} pairs"}), 400

    results = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            return jsonify({"error": f"pairs[{index}] must be an object"}), 400
        try:
            resume_text = validate_text(pair.get("resume_text"), f"pairs[{index}].resume_text")
            job_description = validate_text(
                pair.get("job_description"), f"pairs[{index}].job_description"
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        result = calculate_match(resume_text, job_description)
        save_history(result)
        results.append(result)

    return jsonify({"total": len(results), "results": results})

@app.get("/api/skills/database")
def get_skill_database():
    return jsonify(SKILL_DATABASE)

@app.get("/api/history")
def get_history():
    import json
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit or 20, 100))

    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT match_id, match_score, recommendation,
                   matched_skills, missing_skills, created_at
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    analyses = []
    for row in rows:
        analyses.append({
            "match_id": row["match_id"],
            "match_score": row["match_score"],
            "recommendation": row["recommendation"],
            "matched_skills": json.loads(row["matched_skills"]),
            "missing_skills": json.loads(row["missing_skills"]),
            "processed_at": row["created_at"],
        })
    return jsonify({"total_returned": len(analyses), "analyses": analyses})

@app.delete("/api/history")
def clear_history():
    with database_connection() as connection:
        connection.execute("DELETE FROM analysis_history")
    return jsonify({"message": "Analysis history cleared"})


# Optional semantic matching uses sentence-transformers embeddings.
# The dependency and model are loaded only when this endpoint is requested.
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Semantic embeddings are optional. Install them with: "
                "pip install -r requirements-embeddings.txt"
            ) from error
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

def build_skill_recommendations(missing_skills: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Create transparent, rule-based next steps from recognized skill gaps."""
    recommendations = []
    for item in missing_skills:
        skill = item["skill"]
        category = item["category"]
        recommendations.append({
            "skill": skill,
            "priority": "high" if item["priority"] >= 2.5 else "medium" if item["priority"] >= 1.5 else "low",
            "reason": f"{skill} is listed in the job description under {category.replace('_', ' ')}.",
            "next_step": f"Study {skill}, build a small hands-on implementation, and add a verifiable project example."
        })
    return recommendations

@app.post("/api/match/semantic")
def semantic_match_resume_to_job():
    """Compare a resume and job description with sentence embeddings."""
    data = request.get_json(silent=True) or {}
    try:
        resume_text = validate_text(data.get("resume_text"), "resume_text")
        job_description = validate_text(data.get("job_description"), "job_description")
        model = get_embedding_model()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    import numpy as np
    vectors = model.encode([resume_text, job_description], normalize_embeddings=True)
    similarity = float(np.dot(vectors[0], vectors[1]))
    similarity = max(-1.0, min(1.0, similarity))
    result = calculate_match(resume_text, job_description)
    result["semantic_similarity"] = round(similarity * 100, 2)
    result["semantic_model"] = "sentence-transformers/all-MiniLM-L6-v2"
    result["recommendations"] = build_skill_recommendations(result["missing_skills"])
    return jsonify(result)

@app.post("/api/rank/resumes")
def rank_resumes_for_job():
    """Rank multiple resume texts against one job description."""
    data = request.get_json(silent=True) or {}
    try:
        job_description = validate_text(data.get("job_description"), "job_description")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    resumes = data.get("resumes")
    if not isinstance(resumes, list) or not resumes:
        return jsonify({"error": "resumes must be a non-empty array"}), 400
    if len(resumes) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Maximum number of resumes is {MAX_BATCH_SIZE}"}), 400

    ranked = []
    for index, item in enumerate(resumes):
        if isinstance(item, dict):
            resume_text = item.get("resume_text")
            candidate_id = str(item.get("candidate_id", index + 1))
        else:
            resume_text = item
            candidate_id = str(index + 1)
        try:
            resume_text = validate_text(resume_text, f"resumes[{index}].resume_text")
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        result = calculate_match(resume_text, job_description)
        result["candidate_id"] = candidate_id
        result["recommendations"] = build_skill_recommendations(result["missing_skills"])
        ranked.append(result)

    ranked.sort(key=lambda item: (-item["match_score"], item["candidate_id"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return jsonify({
        "total": len(ranked),
        "ranking_method": "category-weighted recognized-skill overlap",
        "results": ranked
    })

@app.post("/api/recommendations")
def get_skill_gap_recommendations():
    data = request.get_json(silent=True) or {}
    try:
        resume_text = validate_text(data.get("resume_text"), "resume_text")
        job_description = validate_text(data.get("job_description"), "job_description")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    result = calculate_match(resume_text, job_description)
    return jsonify({
        "match_score": result["match_score"],
        "recommendation": result["recommendation"],
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
        "learning_recommendations": build_skill_recommendations(result["missing_skills"]),
        "method": "rule-based recommendations from the configured skill taxonomy and category weights"
    })

initialize_database()

if __name__ == "__main__":
    # Debug mode is disabled by default for safer local demonstrations.
    app.run(host="127.0.0.1", port=5002, debug=False)
