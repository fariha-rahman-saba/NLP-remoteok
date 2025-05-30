import requests
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK is prepared
nltk.download('punkt')
nltk.download('stopwords')

# Load BERT model once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.isalnum() and t not in stop_words]

def fetch_remoteok_jobs():
    url = "https://remoteok.com/api"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        jobs = response.json()[1:]  # Skip first metadata element
        return [{
            'title': job.get('position') or job.get('title'),
            'company': job.get('company', ''),
            'description': job.get('description') or job.get('tags', ''),
            'url': job.get('url', '')
        } for job in jobs if job.get('position') or job.get('title')]
    else:
        return []

def match_resume_with_jobs(resume_text):
    resume_embedding = get_embedding(resume_text)
    matched_jobs = []

    jobs = fetch_remoteok_jobs()  # must return list of dicts!

    for job in jobs:
        title = job.get('title')
        if not isinstance(title, str) or not title.strip():
            continue
        job_embedding = get_embedding(title)
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        if similarity > 0.4:  # Adjust threshold as needed
            matched_jobs.append({
                "title": title,
                "company": job.get('company', ''),
                "description": job.get('description', ''),
                "url": job.get('url', ''),
                "score": round(similarity * 100, 2)
            })

    return matched_jobs
