from pathlib import Path

from django.test import TestCase
from django.contrib.auth import get_user_model
import os
import django
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'AI_Backend.settings'
django.setup()  # Khởi động Django


class MyUserModelTest(TestCase):

    def setUp(self):
        self.user_model = get_user_model()
        self.user = self.user_model.objects.create_user(
            email='testuser@example.com',
            password='testpassword123',
            username='testuser'
        )

    def test_user_creation(self):
        self.assertEqual(self.user.email, 'testuser@example.com')
        self.assertTrue(self.user.check_password('testpassword123'))
        self.assertEqual(self.user.username, 'testuser')
        self.assertTrue(self.user.is_active)

    def test_user_login(self):
        login = self.client.login(email='testuser@example.com', password='testpassword123')
        self.assertTrue(login)

    def test_user_required_fields(self):
        with self.assertRaises(ValueError):
            self.user_model.objects.create_user(email='', password='testpassword123')
        with self.assertRaises(ValueError):
            self.user_model.objects.create_user(email=None, password='testpassword123')

    def test_str_method(self):
        self.assertEqual(str(self.user), 'testuser@example.com')
