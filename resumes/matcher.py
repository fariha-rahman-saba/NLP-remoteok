from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure NLTK is ready
nltk.download('punkt')
nltk.download('stopwords')

# Load BERT tokenizer and model only once
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

def match_resume_with_job_titles(resume_text, job_titles):
    resume_embedding = get_embedding(resume_text)
    resume_tokens = clean_text(resume_text)

    matched = []
    for title in job_titles:
        if not isinstance(title, str) or not title.strip():
            continue  # skip invalid titles
        job_embedding = get_embedding(title)
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        if similarity > 0.4:  # Adjust threshold if needed
            matched.append({
                "title": title,
                "score": round(similarity * 100, 2)
            })

    return matched
