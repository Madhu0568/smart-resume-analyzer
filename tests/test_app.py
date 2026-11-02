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

    def test_tfidf_returns_dictionary(self):
        result = analyzer.extract_tfidf_keywords(
            "Python is useful for data analysis. SQL is used for querying databases."
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
