from django.contrib.auth.base_user import BaseUserManager
from wheel.metadata import _
from django.contrib.auth.models import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from djongo import models

from .app_models.photos import FacePhoto, OCRPhoto, ObjectDetPhoto


class CustomUserManager(BaseUserManager):
    """
    Custom user model manager where email is the unique identifiers
    for authentication instead of usernames.
    """
    def create_user(self, email, password, **extra_fields):
        """
        Create and save a User with the given email and password.
        """
        if not email:
            raise ValueError(_('The Email must be set'))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user


class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(_('email address'), unique=True)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)

    ocr_image_ids = models.JSONField(models.CharField(max_length=24), default=list)
    face_image_ids = models.JSONField(models.CharField(max_length=24), default=list)
    ob_det_image_ids = models.JSONField(models.CharField(max_length=24), default=list)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email

    def image_add(self, images, photo_type):
        if photo_type == 'face':
            self.face_image_ids.extend(images)
        elif photo_type == 'ocr':
            self.ocr_image_ids.extend(images)
        elif photo_type == 'ob_det':
            self.ob_det_image_ids.extend(images)
        self.save()

    def get_image(self, photo_type):
        if photo_type == 'face':
            return self.face_image_ids
        elif photo_type == 'ocr':
            return self.ocr_image_ids
        elif photo_type == 'ob_det':
            return self.ob_det_image_ids
        raise ValueError(f"Unknown photo type: {photo_type}")