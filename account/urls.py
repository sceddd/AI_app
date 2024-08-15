from django.urls import path
from .views import register_user,task_status,upload,login_user

urlpatterns = [
    path('register/', register_user, name='register'),
    path('task_status/<str:task_id>/', task_status, name='task_status'),
    path('login/', login_user, name='login'),
    path('upload/', upload, name='upload'),
]


