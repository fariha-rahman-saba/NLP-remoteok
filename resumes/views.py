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
from .matcher import match_resume_with_jobs
from django.core.exceptions import ValidationError
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
for res in ('punkt', 'stopwords'):
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

def extract_text_from_resume(file):
    try:
        logger.info(f"Extracting text from file: {file.name}")
        file.seek(0)

        if file.name.endswith('.pdf'):
            logger.info("Processing PDF file")
            text = pdfminer.high_level.extract_text(io.BytesIO(file.read()))
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        elif file.name.endswith('.docx'):
            logger.info("Processing DOCX file")
            doc = docx.Document(io.BytesIO(file.read()))
            text = '\n'.join([p.text for p in doc.paragraphs])
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        elif file.name.endswith('.txt'):
            logger.info("Processing TXT file")
            text = file.read().decode('utf-8', errors='ignore')
            logger.info(f"Extracted {len(text)} characters from TXT")
            return text
        else:
            raise ValidationError("Unsupported file format. Please upload PDF, DOCX, or TXT files.")
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise ValidationError(f"Error extracting text: {str(e)}")

def upload_resume_view(request):
    if request.method == 'POST' and request.FILES.get('resume'):
        try:
            resume_file = request.FILES['resume']
            logger.info(f"Processing uploaded file: {resume_file.name}")
            
            resume_text = extract_text_from_resume(resume_file)
            logger.info(f"Successfully extracted text from resume")
            
            matched_jobs = match_resume_with_jobs(resume_text)
            logger.info(f"Found {len(matched_jobs)} matching jobs")
            
            return render(request, 'resumes/result.html', {
                'jobs': matched_jobs,
                'resume': resume_text,
            })
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return render(request, 'resumes/upload.html', {'error': str(e)})
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return render(request, 'resumes/upload.html', {'error': 'An unexpected error occurred.'})

    return render(request, 'resumes/upload.html')

class ResumeUploadView(APIView):
    def post(self, request):
        try:
            file = request.FILES['file']
            logger.info(f"Processing API upload: {file.name}")
            
            text = extract_text_from_resume(file)
            logger.info("Successfully extracted text from resume")
            
            skills = self.extract_skills(text)
            logger.info(f"Extracted skills: {skills}")

            resume = Resume.objects.create(
                name=request.data.get('name', 'Unknown'),
                email=request.data.get('email', ''),
                phone=request.data.get('phone', ''),
                skills=', '.join(skills),
                experience=text[:500],  # Store first 500 chars as experience
                file=file
            )
            return Response({'message': 'Resume uploaded successfully', 'resume': ResumeSerializer(resume).data})
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return Response({'error': str(e)}, status=400)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response({'error': 'An unexpected error occurred'}, status=500)

    def extract_skills(self, text):
        skills_keywords = ["Python", "Django", "React", "Machine Learning", "NLP"]
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return [skill for skill in skills_keywords if skill.lower() in filtered_tokens]

def match_skills(request):
    matched_jobs = []
    if request.method == 'POST':
        try:
            resume_file = request.FILES.get('resume')
            if resume_file:
                logger.info(f"Processing resume for skill matching: {resume_file.name}")
                resume_text = extract_text_from_resume(resume_file)
                matched_jobs = match_resume_with_jobs(resume_text)
                logger.info(f"Found {len(matched_jobs)} matching jobs")
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return render(request, 'resumes/upload_resume.html', {'error': str(e)})
    
    return render(request, 'resumes/upload_resume.html', {
        'matched_jobs': matched_jobs
    })

def show_scraped_jobs(request):
    if request.method == 'POST':
        try:
            keyword = request.POST.get('keyword', 'python developer')
            location = request.POST.get('location', 'Bangladesh')
            logger.info(f"Scraping jobs for keyword: {keyword}, location: {location}")
            jobs = scrape_linkedin_jobs(keyword, location)
            logger.info(f"Found {len(jobs)} jobs")
            return render(request, 'resumes/scraped_jobs.html', {'jobs': jobs})
        except Exception as e:
            logger.error(f"Error scraping jobs: {str(e)}")
            return render(request, 'resumes/scraped_jobs.html', {'error': 'Failed to fetch jobs'})
    
    return render(request, 'resumes/scraped_jobs.html', {'jobs': []})