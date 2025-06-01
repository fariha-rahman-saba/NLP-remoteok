import requests
import json
from datetime import datetime

def scrape_remoteok_jobs(keyword="python", location=None):
    """
    Scrape jobs from RemoteOK API
    """
    url = "https://remoteok.com/api"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        jobs = response.json()[1:]  # Skip first metadata element
        
        filtered_jobs = []
        for job in jobs:
            # Skip jobs without required fields
            if not job.get('position') and not job.get('title'):
                continue
                
            # Convert tags to string if it's a list
            tags = job.get('tags', [])
            if isinstance(tags, list):
                tags = ' '.join(tags)
            
            # Get description, fallback to tags if no description
            description = job.get('description', '')
            if not description and tags:
                description = tags
            
            # Format date
            date = job.get('date', '')
            if date:
                try:
                    date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    date = date_obj.strftime('%Y-%m-%d')
                except:
                    date = ''
            
            filtered_jobs.append({
                'title': job.get('position') or job.get('title', ''),
                'company': job.get('company', ''),
                'description': description,
                'url': job.get('url', ''),
                'tags': tags,
                'date': date,
                'location': job.get('location', 'Remote'),
                'salary': job.get('salary', '')
            })
        
        return filtered_jobs
    except Exception as e:
        print(f"Error fetching jobs: {str(e)}")
        return []

def scrape_linkedin_jobs(keyword="python developer", location="Bangladesh"):
    """
    This function is kept for backward compatibility but now uses RemoteOK instead
    """
    return scrape_remoteok_jobs(keyword, location)
