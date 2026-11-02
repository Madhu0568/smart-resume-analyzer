# Smart Resume Analyzer & Job Matcher

A Python + Flask application that compares resume text with a job description, extracts skills, calculates a category-weighted match score, identifies skill gaps, ranks keywords using TF-IDF, and stores analysis history in SQLite.

## Features

- Skill extraction across programming, frontend, backend, databases, cloud/DevOps, data science/ML, and tools.
- Alias-aware matching (for example, PostgreSQL / Postgres and Node.js / NodeJS).
- Correct boundary handling for names such as C++, C#, and CI/CD.
- Resume section detection using normalized section headings.
- TF-IDF keyword extraction using scikit-learn. Text is split into sentence/line chunks so IDF is calculated from document frequency within the supplied text.
- Weighted match score based on the job description's recognized skills.
- Matched, missing, and additional skill reports.
- Batch matching API.
- SQLite-backed persistent history.
- Browser dashboard.
- Request validation and useful HTTP error responses.

## Requirements

- Python 3.10+
- pip

## Setup

```bash
# 1. Open this project folder in VS Code
# 2. Create a virtual environment
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the application
python app.py
```

Open http://127.0.0.1:5002 in your browser.

The SQLite database `analysis_history.db` is created automatically the first time the app starts. It is local runtime data and should not be committed to GitHub.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/analyze/resume` | Extract resume skills, sections, and keywords |
| POST | `/api/analyze/job` | Extract job skills and keywords |
| POST | `/api/match` | Compare one resume and job description |
| POST | `/api/match/batch` | Compare multiple resume/job pairs |
| GET | `/api/skills/database` | Return supported skill taxonomy |
| GET | `/api/history` | Return recent saved analyses |
| DELETE | `/api/history` | Clear saved analysis history |

## Example request

```bash
curl -X POST http://127.0.0.1:5002/api/match \
  -H "Content-Type: application/json" \
  -d "{\"resume_text\":\"Python SQL Pandas Flask PostgreSQL C++\",\"job_description\":\"Python SQL Pandas Power BI Statistics\"}"
```

## Scoring method

The score is:

`(sum of weights for matched required skills / sum of weights for all recognized required skills) * 100`

Category weights:
- Programming: 3.0
- Backend: 2.5
- Cloud/DevOps: 2.5
- Databases: 2.0
- Data science/ML: 2.0
- Frontend: 1.5
- Tools: 1.0

This is a transparent rule-based score, not a prediction of whether an employer will hire a candidate.

## TF-IDF method

The app uses scikit-learn's `TfidfVectorizer`. It splits a text into sentence/line chunks and treats those chunks as documents. TF-IDF ranks terms using term frequency and inverse document frequency across those chunks. It is a keyword-ranking feature; it is not a semantic language model.

## Important limitations

- The app accepts pasted text; it does not extract text from PDF/DOCX files.
- Skill extraction is taxonomy/dictionary based. A skill not included in the taxonomy may not be detected.
- Match score depends on the predefined category weights and recognized skills. It is an indicator, not a hiring probability.
- Section detection expects recognizable headings.
- Keyword ranking depends on sentence/line segmentation and the supplied text.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Git hygiene

Add the following to `.gitignore` so local runtime files are not committed:

```gitignore
venv/
__pycache__/
*.pyc
analysis_history.db
.env
```
