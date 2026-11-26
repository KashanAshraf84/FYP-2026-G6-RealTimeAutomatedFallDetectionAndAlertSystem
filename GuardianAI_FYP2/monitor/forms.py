from django import forms
from .models import Camera, DetectionSettings, Event


class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = ["name", "location", "kind", "enabled"]


class SettingsForm(forms.ModelForm):
    class Meta:
        model = DetectionSettings
        fields = [
            "confidence",
            "consecutive_frames",
            "cooldown_seconds",
            "email_enabled",
            "email_recipient",
        ]

    def clean(self):
        data = super().clean()
        if data.get("email_enabled") and not data.get("email_recipient"):
            self.add_error("email_recipient", "Enter a recipient to enable email.")
        return data


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Event
        fields = ["status", "notes"]
        widgets = {"notes": forms.Textarea(attrs={"rows": 3, "maxlength": 4000})}

    def clean_notes(self):
        value = self.cleaned_data["notes"]
        if len(value) > 4000:
            raise forms.ValidationError("Use at most 4000 characters.")
        return value
