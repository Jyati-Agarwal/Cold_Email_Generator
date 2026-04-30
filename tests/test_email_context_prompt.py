import unittest

from services.email_generator import _build_user_prompt, _context_links

try:
    from services.resume.pipeline import _valid_portfolio_url
except ModuleNotFoundError:
    _valid_portfolio_url = None


class EmailContextPromptTests(unittest.TestCase):
    def test_prompt_prioritizes_experience_context(self):
        context = {
            "candidate_name": "Jane Doe",
            "candidate_email": "jane@example.com",
            "candidate_phone": "+1 555 0100",
            "role_applied_for": "Full Stack Developer",
            "years_of_experience": "about 2 years",
            "current_or_recent_role": {
                "company": "TechCorp",
                "title": "Software Engineer",
                "duration": "2022-2024",
                "summary": "Built production web applications.",
            },
            "experience_summary": "Jane has about 2 years of full-stack experience at TechCorp.",
            "company_experience": [
                {
                    "company": "TechCorp",
                    "role": "Software Engineer",
                    "duration": "2022-2024",
                    "impact": "Built production web applications.",
                }
            ],
        }

        prompt = _build_user_prompt(context)

        self.assertIn("years_of_experience", prompt)
        self.assertIn("current_or_recent_role", prompt)
        self.assertIn("TechCorp", prompt)

    def test_signature_links_are_limited_to_core_profiles(self):
        context = {
            "application_links": [
                {"label": "LinkedIn", "url": "https://linkedin.com/in/janedoe"},
                {"label": "GitHub", "url": "https://github.com/janedoe"},
                {"label": "Portfolio", "url": "https://janedoe.dev"},
                {"label": "Link", "url": "https://github.com/janedoe/project"},
            ]
        }

        links = _context_links(context)

        self.assertEqual(len(links), 3)
        self.assertNotIn(
            "https://github.com/janedoe/project",
            [link["url"] for link in links],
        )

    def test_github_repo_is_not_portfolio(self):
        if _valid_portfolio_url is None:
            self.skipTest("PyMuPDF is not installed in this local environment")
        self.assertIsNone(
            _valid_portfolio_url("https://github.com/janedoe/project")
        )
        self.assertEqual(
            _valid_portfolio_url("https://janedoe.dev"),
            "https://janedoe.dev",
        )


if __name__ == "__main__":
    unittest.main()
