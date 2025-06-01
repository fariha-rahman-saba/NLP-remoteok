from django.apps import AppConfig


class ResumesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'resumes'
    verbose_name = 'Resume Matching'

    def ready(self):
        try:
            import resumes.templatetags.resume_filters  # noqa
        except ImportError:
            pass
