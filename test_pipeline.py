import os
import requests
from fpdf import FPDF

# 1. Create Dummy PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="John Doe", ln=1, align="C")
pdf.cell(200, 10, txt="john.doe@example.com | github.com/johndoe", ln=2, align="C")
pdf.cell(200, 10, txt="Skills: Python, FastAPI, Next.js, React", ln=3, align="L")
pdf.cell(200, 10, txt="Experience: Senior Software Engineer at TechCorp (2020 - Present)", ln=4, align="L")

# Ensure tests dir exists
os.makedirs("tests", exist_ok=True)
pdf_path = "tests/dummy_resume.pdf"
pdf.output(pdf_path)
print(f"Created dummy PDF at {pdf_path}")

# 2. Test /api/process Endpoint
url = "http://127.0.0.1:8001/api/process"

# Fake Job Description
jd_text = """
We are looking for a Senior Full Stack Engineer.
Must have experience with Python, FastAPI, and React/Next.js.
You will be joining our dynamic team to build AI-powered tools.
Company: OpenAI
Role: Senior Engineer
Please reach out to hiring@openai.com.
"""

print(f"\nSending POST request to {url}...")
try:
    with open(pdf_path, "rb") as f:
        files = {
            "resume": ("dummy_resume.pdf", f, "application/pdf")
        }
        data = {
            "job_description_text": jd_text
        }
        
        response = requests.post(url, files=files, data=data)
        
        print(f"\nResponse Status Code: {response.status_code}")
        try:
            json_resp = response.json()
            import json
            print("\nResponse JSON:")
            print(json.dumps(json_resp, indent=2))
        except Exception as e:
            print("\nResponse Text (Not JSON):")
            print(response.text)
            
except Exception as e:
    print(f"Request failed: {str(e)}")
