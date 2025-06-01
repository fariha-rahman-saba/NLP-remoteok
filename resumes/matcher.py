import requests
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import re
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK is prepared
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize BERT model and tokenizer
try:
    logger.info("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set model to evaluation mode
    logger.info("BERT model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load BERT model: {str(e)}")
    raise

def get_embedding(text, max_length=512):
    try:
        # Clean and preprocess text
        text = clean_text_for_embedding(text)
        logger.info(f"Cleaned text length: {len(text)}")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def clean_text_for_embedding(text):
    """Clean text for BERT embedding"""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text):
    """Extract skills from text using a predefined list of common tech skills"""
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node.js", "django", "flask", "spring", "express", "mongodb", "postgresql",
        "mysql", "aws", "azure", "gcp", "docker", "kubernetes", "machine learning",
        "ai", "data science", "devops", "ci/cd", "git", "agile", "scrum",
        "rest api", "graphql", "microservices", "cloud", "security", "testing",
        "frontend", "backend", "fullstack", "mobile", "ios", "android", "swift",
        "kotlin", "php", "ruby", "rails", "go", "rust", "c++", "c#", ".net"
    ]
    
    text = text.lower()
    found_skills = []
    
    for skill in common_skills:
        if skill in text:
            found_skills.append(skill)
    
    logger.info(f"Found skills: {found_skills}")
    return found_skills

def match_resume_with_jobs(resume_text, threshold=0.0):  # Set threshold to 0 to show all jobs
    try:
        logger.info("Starting job matching process...")
        
        # Extract skills from resume
        resume_skills = extract_skills(resume_text)
        logger.info(f"Extracted {len(resume_skills)} skills from resume")
        
        # Get resume embedding
        logger.info("Generating resume embedding...")
        resume_embedding = get_embedding(resume_text)
        
        # Fetch jobs from RemoteOK
        logger.info("Fetching jobs from RemoteOK...")
        jobs = fetch_remoteok_jobs()
        logger.info(f"Fetched {len(jobs)} jobs from RemoteOK")
        
        if not jobs:
            logger.warning("No jobs fetched from RemoteOK")
            return []

        matched_jobs = []
        for idx, job in enumerate(jobs):
            logger.info(f"Processing job {idx + 1}/{len(jobs)}: {job.get('title', '')}")
            
            # Combine title, description, and tags for better matching
            job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('tags', '')}"
            
            # Get job embedding
            job_embedding = get_embedding(job_text)
            
            # Calculate similarity
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            logger.info(f"Semantic similarity score: {similarity:.3f}")
            
            # Extract skills from job description
            job_skills = extract_skills(job_text)
            logger.info(f"Found {len(job_skills)} skills in job description")
            
            # Calculate skill match percentage
            matching_skills = set(resume_skills) & set(job_skills)
            skill_match_percentage = len(matching_skills) / len(job_skills) if job_skills else 0
            logger.info(f"Skill match percentage: {skill_match_percentage:.3f}")
            
            # Combine similarity score with skill match
            final_score = (similarity * 0.7) + (skill_match_percentage * 0.3)
            logger.info(f"Final score: {final_score:.3f}")
            
            # Add all jobs with their scores
            matched_jobs.append({
                "title": job.get('title', ''),
                "company": job.get('company', ''),
                "description": job.get('description', ''),
                "url": job.get('url', ''),
                "score": round(final_score * 100, 2),
                "matching_skills": list(matching_skills),
                "required_skills": list(job_skills),
                "location": job.get('location', 'Remote'),
                "salary": job.get('salary', ''),
                "date": job.get('date', '')
            })

        # Sort by score and date
        matched_jobs.sort(key=lambda x: (-x['score'], x['date'] if x['date'] else ''))
        
        # Return top 20 jobs
        top_jobs = matched_jobs[:20]
        logger.info(f"Returning top {len(top_jobs)} jobs")
        return top_jobs

    except Exception as e:
        logger.error(f"Error in job matching: {str(e)}")
        return []

def clean_job_description(description):
    """Clean job description by removing HTML tags and extra whitespace"""
    if not description:
        return ""
    
    # Remove HTML tags
    soup = BeautifulSoup(description, 'html.parser')
    text = soup.get_text()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()

def fetch_remoteok_jobs():
    """Fetch jobs from RemoteOK API"""
    url = "https://remoteok.com/api"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        logger.info("Fetching jobs from RemoteOK API...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        jobs = response.json()[1:]  # Skip first metadata element
        logger.info(f"Successfully fetched {len(jobs)} jobs from RemoteOK")
        
        filtered_jobs = []
        for job in jobs:
            if job.get('position') or job.get('title'):
                # Clean the description
                description = clean_job_description(job.get('description', ''))
                if not description and job.get('tags'):
                    description = ' '.join(job.get('tags', []))
                
                filtered_jobs.append({
                    'title': job.get('position') or job.get('title', ''),
                    'company': job.get('company', ''),
                    'description': description,
                    'url': job.get('url', ''),
                    'tags': job.get('tags', []),
                    'location': job.get('location', 'Remote'),
                    'salary': job.get('salary', ''),
                    'date': job.get('date', '')
                })
        
        logger.info(f"Filtered to {len(filtered_jobs)} valid jobs")
        return filtered_jobs
    except Exception as e:
        logger.error(f"Error fetching jobs from RemoteOK: {str(e)}")
        return []
