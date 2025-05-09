from django.db import models

class Resume(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True, null=True)
    skills = models.TextField()
    experience = models.TextField()
    file = models.FileField(upload_to='resumes/')

    def __str__(self):
        return self.name