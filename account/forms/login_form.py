from django.contrib.auth import authenticate

from ..models import CustomUser

from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.utils.translation import gettext_lazy as _


class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(label=_('Email'), max_length=254)

