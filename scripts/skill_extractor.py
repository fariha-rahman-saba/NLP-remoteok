import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample resume and job description for testing
resume_text = """
Experienced Software Developer with expertise in Python, Django, Machine Learning, and Data Science. Strong background in web development and software architecture.
"""
job_description = """
We are looking for a Software Engineer with experience in Python, JavaScript, and AI technologies. Knowledge of Django, Flask, and cloud computing is a plus.
"""

# Define a simple list of skills (can be expanded)
skills = ["python", "django", "java", "javascript", "flask", "machine learning", "data science", "cloud computing"]

# Function for text pre-processing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text into words
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Function to extract skills from text
def extract_skills(text, skills):
    tokens = preprocess_text(text)
    extracted_skills = set()
    
    for word in tokens:
        if word in skills:
            extracted_skills.add(word)
    
    return extracted_skills

# Extract skills from resume and job description
extracted_skills_resume = extract_skills(resume_text, skills)
extracted_skills_job_desc = extract_skills(job_description, skills)

print("Extracted Skills from Resume:", extracted_skills_resume)
print("Extracted Skills from Job Description:", extracted_skills_job_desc)
