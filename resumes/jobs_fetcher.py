import requests

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
