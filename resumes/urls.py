from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_resume_view, name='upload_resume'),
    path('match_skills/', views.match_skills, name='match_skills'),
    path('scraped-jobs/', views.show_scraped_jobs, name='scraped_jobs'),
]