import unittest
import tempfile
from pathlib import Path

import app as analyzer


class ResumeAnalyzerTests(unittest.TestCase):
    def test_special_character_skills(self):
        text = "I use C++, C#, Node.js, PostgreSQL and CI/CD."
        skills = analyzer.extract_skills(text)
        self.assertIn("c++", skills)
        self.assertIn("c#", skills)
        self.assertIn("node.js", skills)
        self.assertIn("postgresql", skills)
        self.assertIn("ci/cd", skills)

    def test_avoids_substring_false_positive_for_sql(self):
        skills = analyzer.extract_skills("I have NoSQL database experience.")
        self.assertNotIn("sql", skills)

    def test_weighted_match(self):
        result = analyzer.calculate_match(
            "Python, SQL, Pandas",
            "Python, SQL, Pandas, Power BI"
        )
        self.assertGreater(result["match_score"], 0)
        self.assertIn("power bi", [item["skill"] for item in result["missing_skills"]])

    def test_sections(self):
        sections = analyzer.extract_sections(
            "Madhuri Katakam\nProfessional Summary\nData analyst\nEducation\nNIT Patna\nTechnical Skills\nPython SQL"
        )
        self.assertIn("summary", sections)
        self.assertIn("education", sections)
        self.assertIn("skills", sections)

    def test_elip_llm_skills_are_recognized(self):
        skills = analyzer.extract_skills("Built RAG pipelines with LangChain, embeddings, vector databases, prompt engineering and tool calling.")
        for skill in ["rag", "langchain", "embeddings", "vector databases", "prompt engineering", "tool calling"]:
            self.assertIn(skill, skills)

    def test_rank_resumes_endpoint_orders_candidates(self):
        client = analyzer.app.test_client()
        response = client.post("/api/rank/resumes", json={
            "job_description": "Python Flask SQL REST APIs Docker",
            "resumes": [
                {"candidate_id": "weaker", "resume_text": "Java Spring"},
                {"candidate_id": "stronger", "resume_text": "Python Flask SQL REST APIs Docker"}
            ]
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["results"][0]["candidate_id"], "stronger")
        self.assertEqual(payload["results"][0]["rank"], 1)

    def test_recommendations_endpoint_returns_skill_gaps(self):
        client = analyzer.app.test_client()
        response = client.post("/api/recommendations", json={
            "resume_text": "Python Flask",
            "job_description": "Python Flask Docker PostgreSQL"
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["learning_recommendations"])

    def test_tfidf_returns_dictionary(self):
        result = analyzer.extract_tfidf_keywords(
            "Python is useful for data analysis. SQL is used for querying databases."
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
