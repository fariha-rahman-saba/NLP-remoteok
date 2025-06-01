from django import forms

class ResumeUploadForm(forms.Form):
    resume = forms.FileField(
        required=True,
        widget=forms.FileInput(attrs={
            'accept': '.pdf,.docx,.txt',
            'class': 'custom-file-input',
            'id': 'resume'
        })
    )

    def clean_resume(self):
        resume = self.cleaned_data.get('resume')
        if not resume:
            raise forms.ValidationError('Please select a resume file to upload.')
        
        # Check file size (5MB limit)
        if resume.size > 5 * 1024 * 1024:
            raise forms.ValidationError('File size too large. Maximum size is 5MB.')
        
        # Check file extension
        ext = resume.name.split('.')[-1].lower()
        if ext not in ['pdf', 'docx', 'txt']:
            raise forms.ValidationError('Unsupported file format. Please upload PDF, DOCX, or TXT files.')
        
        return resume
