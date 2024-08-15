import uuid

from celery.result import AsyncResult
from django.conf import settings
from django.contrib.auth import login, authenticate, get_user_model
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import permission_classes, api_view
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken

from .app_models.photos import Photo
from .forms.login_form import CustomAuthenticationForm
from .forms.register_form import CustomUserCreationForm
from .tasks import batch_upload_images_to_mongodb

User = get_user_model()


@csrf_exempt
def task_status(_, task_id):
    result = AsyncResult(task_id)
    response = {
        'task_id': task_id,
        'state': result.state,
        'result': result.result if result.state == 'SUCCESS' else None,
        'error': str(result.info) if result.state == 'FAILURE' else None,
    }
    return JsonResponse(response)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@login_required
def user_photos(request):
    user = request.user
    photos = Photo.objects.filter(user=user)
    photo_list = [{'image_id': photo.image_id, 'image_name': photo.image_name} for photo in photos]
    return JsonResponse({'photos': photo_list})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
# @permission_classes([AllowAny])
@csrf_exempt
def upload(request):
    if request.method == 'POST' and 'files' in request.FILES:
        img_ids = []

        user = request.user

        redis_client = settings.REDIS_CLIENT
        batch_size = settings.LMDB_BATCH_SIZE

        for image in request.FILES.getlist('files'):
            image_data = image.read()
            image_id = 'img_{}'.format(uuid.uuid1())
            Photo(user=user, image_id=image_id, status=Photo.Status.UPLOADED).save()
            redis_client.set(image_id, image_data)
            img_ids.append(image_id)

        # flag_key = 'processing_flag_{}'.format(uuid.uuid1())
        # redis_client.set(flag_key, 'pending')

        batches = [img_ids[i:i + batch_size] for i in range(0, len(img_ids), batch_size)]

        for batch in batches:
            if batch:
                batch_upload_images_to_mongodb.delay(batch)

        return JsonResponse({'message': 'Tasks created for all batches'}, status=202)

    return JsonResponse({'error': 'No image file found'}, status=400)


@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request,data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                tokens = get_tokens_for_user(user)
                return JsonResponse({"message": "User logged in successfully","tokens": tokens}, status=200)
            else:
                return JsonResponse({"error": "Invalid password"}, status=400)
        else:
            return JsonResponse({"errors": form.errors}, status=400)
    return JsonResponse({"error": "Only POST method is allowed"}, status=405)


@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return JsonResponse({"message": "User created successfully", "token":
                                 get_tokens_for_user(user)}, status=201)
        else:
            return JsonResponse({"errors": form.errors}, status=400)
    return JsonResponse({"error": "Only POST method is allowed"}, status=405)


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }




