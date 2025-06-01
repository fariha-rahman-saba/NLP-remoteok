from django import template

register = template.Library()

@register.filter
def score_color(score):
    """Return a color based on the match score"""
    if score >= 70:
        return '#28a745'  # Green for high matches
    elif score >= 40:
        return '#ffc107'  # Yellow for medium matches
    else:
        return '#dc3545'  # Red for low matches 