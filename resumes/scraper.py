from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def scrape_linkedin_jobs(keyword="python developer", location="Bangladesh"):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    search_url = f"https://www.linkedin.com/jobs/search/?keywords={keyword}&location={location}"
    driver.get(search_url)

    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    jobs = []
    listings = soup.select("ul.jobs-search__results-list li")

    for job in listings:
        title = job.select_one("h3").get_text(strip=True) if job.select_one("h3") else "No title"
        company = job.select_one("h4").get_text(strip=True) if job.select_one("h4") else "No company"
        link_tag = job.find("a", href=True)
        job_url = f"https://www.linkedin.com{link_tag['href']}" if link_tag else "#"

        jobs.append({
            "title": title,
            "company": company,
            "url": job_url
        })

    return jobs
