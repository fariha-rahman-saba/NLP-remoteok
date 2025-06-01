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
    """Extract skills from text using a comprehensive list of tech skills"""
    common_skills = {
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "go", "rust", "swift", "kotlin",
        "scala", "r", "matlab", "perl", "haskell", "elixir", "clojure", "dart",
        
        # Web Development
        "html", "css", "react", "angular", "vue", "svelte", "next.js", "nuxt.js", "gatsby", "jquery", "bootstrap",
        "tailwind", "sass", "less", "webpack", "vite", "rollup", "babel", "es6", "typescript",
        
        # Backend & Frameworks
        "node.js", "express", "django", "flask", "fastapi", "spring", "laravel", "rails", "asp.net", "gin",
        "echo", "fiber", "rocket", "actix", "phoenix", "play", "ktor",
        
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "elasticsearch", "dynamodb", "firebase",
        "neo4j", "couchdb", "mariadb", "oracle", "sqlite", "graphql",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins", "gitlab", "github",
        "circleci", "travis", "heroku", "digitalocean", "cloudflare", "nginx", "apache",
        
        # AI & ML
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn", "numpy", "pandas",
        "opencv", "nltk", "spacy", "huggingface", "transformers", "bert", "gpt", "computer vision",
        "natural language processing", "nlp", "reinforcement learning", "data science",
        
        # Mobile Development
        "ios", "android", "react native", "flutter", "xamarin", "ionic", "cordova", "swift", "kotlin",
        "objective-c", "java android", "mobile development",
        
        # Security
        "cybersecurity", "penetration testing", "ethical hacking", "security", "cryptography", "authentication",
        "authorization",
        
        # Testing & QA
        "unit testing", "integration testing", "e2e testing", "jest", "pytest", "junit", "selenium",
        "cypress", "playwright", "testcafe", "qa", "quality assurance", "automated testing",
        
        # Methodologies & Tools
        "agile", "scrum", "kanban", "devops", "ci/cd", "git", "svn", "jira", "confluence", "trello",
        "asana", "monday", "notion", "figma", "sketch", "adobe xd", "photoshop", "illustrator",
        
        # Architecture & Design
        "microservices", "rest api", "graphql", "soap", "grpc", "websocket", "message queue", "kafka",
        "rabbitmq", "system design", "software architecture", "design patterns", "clean code",
        "solid principles", "ddd", "tdd", "bdd",
        
        # Data & Analytics
        "data analysis", "data visualization", "tableau", "power bi", "looker", "metabase", "superset",
        "etl", "data warehousing", "big data", "hadoop", "spark", "kafka", "airflow", "dbt",
        
        # Blockchain
        "blockchain", "ethereum", "solidity", "web3", "smart contracts", "defi", "nft", "cryptocurrency",
        "bitcoin", "hyperledger", "consensus algorithms",
        
        # Game Development
        "unity", "unreal engine", "game development", "3d modeling", "blender", "maya", "opengl",
        "directx", "game design", "game programming",
        
        # Embedded & IoT
        "embedded systems", "iot", "arduino", "raspberry pi", "fpga", "verilog", "vhdl", "microcontrollers",
        "real-time systems", "firmware", "device drivers"
    }
    
    text = text.lower()
    found_skills = set()
    
    # Direct skill matching with word boundaries
    for skill in common_skills:
        # Use word boundaries to ensure we match whole words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.add(skill)
    
    # Pattern matching for specific variations only
    patterns = {
        r'\b(?:js|javascript)\b': 'javascript',
        r'\b(?:ts|typescript)\b': 'typescript',
        r'\b(?:py|python)\b': 'python',
        r'\b(?:ml|machine learning)\b': 'machine learning',
        r'\b(?:dl|deep learning)\b': 'deep learning',
        r'\b(?:ai|artificial intelligence)\b': 'ai',
        r'\b(?:devops|dev ops)\b': 'devops',
        r'\b(?:ci/cd|ci cd|continuous integration)\b': 'ci/cd',
        r'\b(?:aws|amazon web services)\b': 'aws',
        r'\b(?:gcp|google cloud platform)\b': 'gcp',
        r'\b(?:azure|microsoft azure)\b': 'azure',
        r'\b(?:react|reactjs|react.js)\b': 'react',
        r'\b(?:angular|angularjs|angular.js)\b': 'angular',
        r'\b(?:vue|vuejs|vue.js)\b': 'vue',
        r'\b(?:node|nodejs|node.js)\b': 'node.js',
        r'\b(?:express|expressjs|express.js)\b': 'express',
        r'\b(?:django|djangorest|django rest)\b': 'django',
        r'\b(?:flask|flask-rest|flask rest)\b': 'flask',
        r'\b(?:spring|spring boot|springboot)\b': 'spring',
        r'\b(?:rails|ruby on rails)\b': 'rails',
        r'\b(?:laravel|php laravel)\b': 'laravel',
        r'\b(?:asp|asp.net|dotnet|.net)\b': '.net',
        r'\b(?:postgres|postgresql|postgre)\b': 'postgresql',
        r'\b(?:mongo|mongodb)\b': 'mongodb',
        r'\b(?:redis|cache)\b': 'redis',
        r'\b(?:elastic|elasticsearch)\b': 'elasticsearch',
        r'\b(?:dynamo|dynamodb)\b': 'dynamodb',
        r'\b(?:firebase|firestore)\b': 'firebase',
        r'\b(?:neo4j|graph database)\b': 'neo4j',
        r'\b(?:couch|couchdb)\b': 'couchdb',
        r'\b(?:mariadb|mysql)\b': 'mysql',
        r'\b(?:oracle|oracle db)\b': 'oracle',
        r'\b(?:sqlite|sqlite3)\b': 'sqlite',
        r'\b(?:graphql|gql)\b': 'graphql',
        r'\b(?:docker|container)\b': 'docker',
        r'\b(?:k8s|kubernetes)\b': 'kubernetes',
        r'\b(?:terraform|infrastructure as code)\b': 'terraform',
        r'\b(?:ansible|automation)\b': 'ansible',
        r'\b(?:jenkins|ci server)\b': 'jenkins',
        r'\b(?:gitlab|git lab)\b': 'gitlab',
        r'\b(?:github|git hub)\b': 'github',
        r'\b(?:circleci|circle ci)\b': 'circleci',
        r'\b(?:travis|travis ci)\b': 'travis',
        r'\b(?:heroku|heroku platform)\b': 'heroku',
        r'\b(?:digitalocean|digital ocean)\b': 'digitalocean',
        r'\b(?:cloudflare|cloud flare)\b': 'cloudflare',
        r'\b(?:nginx|web server)\b': 'nginx',
        r'\b(?:apache|web server)\b': 'apache',
        r'\b(?:tensorflow|tf)\b': 'tensorflow',
        r'\b(?:pytorch|torch)\b': 'pytorch',
        r'\b(?:keras|deep learning)\b': 'keras',
        r'\b(?:scikit|scikit-learn)\b': 'scikit-learn',
        r'\b(?:numpy|numerical python)\b': 'numpy',
        r'\b(?:pandas|data analysis)\b': 'pandas',
        r'\b(?:opencv|computer vision)\b': 'opencv',
        r'\b(?:nltk|natural language)\b': 'nltk',
        r'\b(?:spacy|nlp)\b': 'spacy',
        r'\b(?:huggingface|transformers)\b': 'huggingface',
        r'\b(?:bert|transformer)\b': 'bert',
        r'\b(?:gpt|generative)\b': 'gpt',
        r'\b(?:cv|computer vision)\b': 'computer vision',
        r'\b(?:nlp|natural language)\b': 'natural language processing',
        r'\b(?:rl|reinforcement)\b': 'reinforcement learning',
        r'\b(?:ds|data science)\b': 'data science',
        r'\b(?:react native|react-native)\b': 'react native',
        r'\b(?:flutter|mobile)\b': 'flutter',
        r'\b(?:xamarin|mobile)\b': 'xamarin',
        r'\b(?:ionic|mobile)\b': 'ionic',
        r'\b(?:cordova|phonegap)\b': 'cordova',
        r'\b(?:swift|ios)\b': 'swift',
        r'\b(?:kotlin|android)\b': 'kotlin',
        r'\b(?:objective-c|objc)\b': 'objective-c',
        r'\b(?:java android|android)\b': 'java android',
        r'\b(?:mobile dev|mobile development)\b': 'mobile development',
        r'\b(?:cyber|cybersecurity)\b': 'cybersecurity',
        r'\b(?:pentest|penetration)\b': 'penetration testing',
        r'\b(?:hacking|ethical)\b': 'ethical hacking',
        r'\b(?:sec|security)\b': 'security',
        r'\b(?:crypto|cryptography)\b': 'cryptography',
        r'\b(?:auth|authentication)\b': 'authentication',
        r'\b(?:authz|authorization)\b': 'authorization',
        r'\b(?:oauth|openid)\b': 'oauth',
        r'\b(?:jwt|json web token)\b': 'jwt',
        r'\b(?:ssl|tls)\b': 'ssl',
        r'\b(?:encrypt|encryption)\b': 'encryption',
        r'\b(?:fw|firewall)\b': 'firewall',
        r'\b(?:vpn|virtual private)\b': 'vpn',
        r'\b(?:unit test|unit testing)\b': 'unit testing',
        r'\b(?:integration test|integration testing)\b': 'integration testing',
        r'\b(?:e2e|end to end)\b': 'e2e testing',
        r'\b(?:automated test|automated testing)\b': 'automated testing'
    }
    
    for pattern, skill in patterns.items():
        if re.search(pattern, text):
            found_skills.add(skill)
    
    logger.info(f"Found skills: {found_skills}")
    return list(found_skills)

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
            
            # Calculate required skills percentage
            required_skills_percentage = len(job_skills) / 20 if job_skills else 0  # Assuming max 20 required skills
            logger.info(f"Required skills percentage: {required_skills_percentage:.3f}")
            
            # Calculate skill coverage
            skill_coverage = len(matching_skills) / max(len(resume_skills), 1)
            logger.info(f"Skill coverage: {skill_coverage:.3f}")
            
            # Combine scores with new weights:
            # - 70% skill matching (importance of matching required skills)
            # - 20% semantic similarity (context and related skills)
            # - 10% skill coverage (how many of your skills are utilized)
            final_score = (
                (skill_match_percentage * 0.7) +  # How well you match the required skills
                (similarity * 0.2) +              # Semantic similarity for context
                (skill_coverage * 0.1)            # How many of your skills are used
            )
            
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
                "date": job.get('date', ''),
                "skill_match_percentage": round(skill_match_percentage * 100, 2),
                "semantic_similarity": round(similarity * 100, 2),
                "skill_coverage": round(skill_coverage * 100, 2)
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
