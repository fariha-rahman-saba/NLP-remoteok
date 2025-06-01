# Resume Job Matcher

A web application that matches your resume with job listings. It analyzes your resume and finds the best matching jobs based on skills, experience, and job requirements.

## Features

- Upload your resume (PDF, DOCX, or TXT)
- Automatic skill extraction from your resume
- Job matching based on skills and experience
- Match score with detailed breakdown
- Beautiful and user-friendly interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume-job-matcher
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py migrate
```

## Running the Project

1. Start the development server:
```bash
python manage.py runserver
```

2. Open your browser and go to:
```
http://localhost:8000
```

3. Upload your resume and find matching jobs!

## Requirements

- Python 3.8 or higher
- Django 4.2 or higher
- Other dependencies listed in requirements.txt 