import unittest

from services.resume.text_fallback import extract_links_from_text


class ResumeExtractionTests(unittest.TestCase):
    def test_extracts_resume_contact_details_and_links(self):
        text = """
        Jane Doe
        New Delhi | +91 98765 43210 | jane.doe@example.com
        LinkedIn: linkedin.com/in/janedoe
        GitHub: github.com/janedoe
        Portfolio: janedoe.dev
        Project: https://github.com/janedoe/searchbot
        Skills: Next.js, Node.js, ASP.NET, Socket.io
        """

        result = extract_links_from_text(text)

        self.assertEqual(result["email"], "jane.doe@example.com")
        self.assertEqual(result["phone"], "+91 98765 43210")
        self.assertEqual(result["linkedin"], "https://linkedin.com/in/janedoe")
        self.assertEqual(result["github"], "https://github.com/janedoe")
        self.assertEqual(result["portfolio"], "https://janedoe.dev")
        self.assertIn("https://github.com/janedoe/searchbot", result["other_links"])
        self.assertNotIn("https://Next.js", result["all_links"])
        self.assertNotIn("https://Node.js", result["all_links"])


if __name__ == "__main__":
    unittest.main()
