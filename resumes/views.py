import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Resume
from .serializers import ResumeSerializer
import pdfminer.high_level
import docx
import io
from .forms import ResumeUploadForm
from django.shortcuts import render
from .scraper import scrape_linkedin_jobs
from .scraper import scrape_linkedin_jobs
from django.shortcuts import render
from .matcher import match_resume_with_jobs


for res in ('punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res, quiet=True)


def extract_text_from_resume(file):
    try:
        file.seek(0)

        if file.name.endswith('.pdf'):
            return pdfminer.high_level.extract_text(io.BytesIO(file.read()))
        elif file.name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file.read()))
            return '\n'.join([p.text for p in doc.paragraphs])
        elif file.name.endswith('.txt'):
            return file.read().decode('utf-8', errors='ignore')
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error extracting text: {e}"


def upload_resume_view(request):
    if request.method == 'POST' and request.FILES.get('resume'):
        resume_file = request.FILES['resume']
        resume_text = extract_text_from_resume(resume_file)

        if resume_text.startswith("Error"):
            return render(request, 'resume_matcher/upload.html', {'error': resume_text})

        matched_jobs = match_resume_with_jobs(resume_text)

        return render(request, 'resume_matcher/result.html', {'matches': matched_jobs})

    return render(request, 'resume_matcher/upload.html')

def upload_resume_view(request):
    if request.method == 'POST' and request.FILES.get('resume'):
        resume_file = request.FILES['resume']
        resume_text = resume_file.read().decode('utf-8', errors='ignore')
        matched_jobs = match_resume_with_jobs(resume_text)
        return render(request, 'resumes/result.html', {
            'jobs': matched_jobs,
            'resume': resume_text,
        })

    return render(request, 'resumes/upload.html')

class ResumeUploadView(APIView):
    def post(self, request):
        file = request.FILES['file']
        text = extract_text_from_resume(file)

        if text.startswith("Error") or text == "Unsupported file format":
            return Response({'error': text}, status=400)

        skills = self.extract_skills(text)

        resume = Resume.objects.create(
            name=request.data.get('name', 'Unknown'),
            email=request.data.get('email', ''),
            phone=request.data.get('phone', ''),
            skills=', '.join(skills),
            file=file
        )
        return Response({'message': 'Resume uploaded successfully', 'resume': ResumeSerializer(resume).data})

    def extract_skills(self, text):
        skills_keywords = ["Python", "Django", "React", "Machine Learning", "NLP"]
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return [skill for skill in skills_keywords if skill.lower() in filtered_tokens]


def match_skills(request):
    matched_jobs = []
    if request.method == 'POST':
        resume_file = request.FILES.get('resume')
        if resume_file:
            resume_text = extract_text_from_resume(resume_file)
            matched_jobs = match_resume_with_jobs(resume_text)
    
    return render(request, 'resume_matcher/upload_resume.html', {
        'matched_jobs': matched_jobs
    })

def show_scraped_jobs(request):
    if request.method == 'POST':
        keyword = request.POST.get('keyword', 'python developer')
        location = request.POST.get('location', 'Bangladesh')
        jobs = scrape_linkedin_jobs(keyword, location)
        return render(request, 'resume_matcher/scraped_jobs.html', {'jobs': jobs})
    
    return render(request, 'resume_matcher/scraped_jobs.html', {'jobs': []})