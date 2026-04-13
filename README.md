# Smart Resume Analyzer & Job Matcher

> This project demonstrates backend system design concepts including APIs, data processing, and asynchronous workflows.

I built this after spending too long manually comparing my resume to job descriptions, trying to figure out which skills were missing. The goal was to automate that — parse both documents, extract the skills from each, and tell me exactly what's missing and how important those gaps are.

The interesting part was the matching algorithm. Simple string matching gave too many false negatives (e.g., "PostgreSQL" vs "postgres"). I ended up building a skill taxonomy with 7 categories and weighted scoring — a missing "cloud" skill counts for more than a missing "tool" because that's how most JDs are weighted in practice.

## What it does

- **Resume parsing** — detects sections (experience, education, skills, projects, certifications)
- **Skill extraction** across 7 categories: programming languages, web frontend, web backend, databases, cloud & DevOps, data science/ML, tools
- **Weighted match scoring** — programming/backend/cloud skills are weighted higher than general tools
- **TF-IDF-inspired keyword ranking** — surfaces the most important terms in each document beyond exact skill matches
- **Skill gap report** — missing skills ranked by importance with category context
- **Batch processing** — handles multiple resume-job pairs in a single request
- Processes each request in under 800ms

## Tech Stack

Python · Flask · NLP (regex + TF-IDF) · JSON

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Opens at `http://localhost:5002`. Load sample data from the dashboard to try it immediately.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze/resume` | Extract skills, sections, and keywords from a resume |
| POST | `/api/analyze/job` | Extract required skills from a job description |
| POST | `/api/match` | Match resume against JD — returns score + skill gaps |
| POST | `/api/match/batch` | Batch process multiple resume-JD pairs |
| GET | `/api/skills/database` | View the full skill taxonomy |
| GET | `/api/history` | Recent analysis history |

## Example

```bash
curl -X POST http://localhost:5002/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with Flask, Django, PostgreSQL, AWS, Git experience",
    "job_description": "Looking for Python developer with Django, PostgreSQL, Docker, Kubernetes, CI/CD"
  }'
```

```json
{
  "match_id": "f3a1b2c4",
  "match_score": 68.5,
  "matched_skills": ["python", "django", "postgresql"],
  "missing_skills": [
    {"skill": "docker",     "category": "cloud",      "priority": 2.5},
    {"skill": "kubernetes", "category": "cloud",      "priority": 2.5},
    {"skill": "ci/cd",      "category": "cloud",      "priority": 2.5}
  ],
  "extra_skills": ["flask", "aws", "git"],
  "recommendation": "Good match with some gaps"
}
```

## Scoring

| Category | Weight |
|----------|--------|
| Programming languages | 3.0 |
| Backend frameworks | 2.5 |
| Cloud / DevOps | 2.5 |
| Databases | 2.0 |
| Data science / ML | 2.0 |
| Frontend | 1.5 |
| Tools | 1.0 |

Score = (sum of weights of matched skills) / (sum of weights of required skills) × 100

## Skill Categories Supported

- **Programming**: Python, Java, JavaScript, TypeScript, Go, Rust, C++, and more
- **Backend**: Flask, Django, Express, FastAPI, Spring, Node.js
- **Frontend**: React, Angular, Vue, Next.js, Tailwind
- **Databases**: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
- **Cloud**: AWS, Azure, GCP, Docker, Kubernetes, Terraform, CI/CD
- **Data/ML**: Pandas, NumPy, scikit-learn, TensorFlow, PyTorch
- **Tools**: Git, Postman, Linux, Agile/Scrum
