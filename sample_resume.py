import re

# ===============================
# SMART RESUME ANALYZER & JOB MATCHER
# ===============================

SKILLS_DB = [
    "python", "java", "c", "c++", "sql", "html", "css", "javascript",
    "excel", "communication", "problem solving", "teamwork",
    "data analysis", "machine learning", "git"
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def extract_skills(text):
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return found


def match_skills(resume_skills, job_skills):
    matched = set(resume_skills) & set(job_skills)
    missing = set(job_skills) - set(resume_skills)

    if len(job_skills) == 0:
        percentage = 0
    else:
        percentage = (len(matched) / len(job_skills)) * 100

    return matched, missing, round(percentage, 2)


def main():
    print("=== SMART RESUME ANALYZER ===\n")

    # Resume input
    resume_path = input("Enter resume file path (txt): ")

    try:
        with open(resume_path, "r", encoding="utf-8") as file:
            resume_text = file.read()
    except FileNotFoundError:
        print("Error: Resume file not found.")
        return

    resume_text = clean_text(resume_text)

    # Job description input
    print("\nPaste Job Description (type END to finish):")
    job_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        job_lines.append(line)

    job_text = clean_text(" ".join(job_lines))

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched, missing, percentage = match_skills(resume_skills, job_skills)

    print("\n--- ANALYSIS RESULT ---")
    print(f"Match Percentage: {percentage}%")

    print("\nMatched Skills:")
    if matched:
        for skill in matched:
            print(f"- {skill}")
    else:
        print("None")

    print("\nMissing Skills:")
    if missing:
        for skill in missing:
            print(f"- {skill}")
    else:
        print("None")

    if missing:
        print("\nSuggestions:")
        for skill in missing:
            print(f"- Learn basic {skill}")


if __name__ == "__main__":
    main()
