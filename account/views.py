import logging
import uuid

from celery.result import AsyncResult
from django.conf import settings
from django.contrib.auth import login, authenticate, get_user_model
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import permission_classes, api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken

from .app_models.photos import get_photo_class, AbstractPhoto
from .forms.login_form import CustomAuthenticationForm
from .forms.register_form import CustomUserCreationForm
from .project_utils._utils import chunked_iterable
from .tasks.tasks import batch_upload_images_to_mongodb

User = get_user_model()
logger = logging.getLogger(__name__)


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


def get_photo(photo_type):
    if photo_type not in ['face', 'ocr', 'ob_det']:
        return JsonResponse({'error': 'Invalid photo type'}, status=400)
    return get_photo_class(photo_type)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def user_photos(request):
    if request.method == 'GET':
        photo_type = request.GET.get('type','').lower()
        user = request.user
        photos_id = user.get_image(photo_type)
        print(photos_id)
        photo_class = get_photo(photo_type)
        photos = photo_class.objects.filter(image_id__in=photos_id)
        print(photos)
        return JsonResponse({'photos': [photo.to_dict() for photo in photos]}, status=200)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def upload(request):
    if request.method == 'POST' and 'files' in request.FILES:
        user = request.user
        redis_client = settings.REDIS_CLIENT
        batch_size = settings.LMDB_BATCH_SIZE

        photo_type = request.POST.get('type', '').lower()
        photo_class = get_photo(photo_type)
        files = request.FILES.getlist('files')
        all_img_ids = []
        for file_chunk in chunked_iterable(files, batch_size):
            img_ids = []
            for image in file_chunk:
                image_data = image.read()
                image_id = 'img_{}'.format(uuid.uuid1())
                redis_client.set(image_id, image_data)

                photo = photo_class(image_id=image_id, status=AbstractPhoto.Status.UPLOADED)
                photo.save()

                img_ids.append(image_id)
            batch_upload_images_to_mongodb.apply_async(args=[img_ids,photo_type], queue='image_upload')
            all_img_ids.extend(img_ids)
        user.image_add(image=all_img_ids,photo_type=photo_type)

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


def logout_user(request):
    return JsonResponse({"message": "User logged out successfully"}, status=200)


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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_image_from_gridfs(request, image_id):

    try:
        photo_type = get_photo_class(request.GET.get('type', ''))
        photo = get_object_or_404(photo_type, image_id=image_id)

        if photo.user != request.user:
            return HttpResponse("You do not have permission to access this image.", status=403)

        if not photo.gridfs_id:
            return HttpResponse("Image not found in GridFS.", status=404)

        image_data = photo.get_image_from_gridfs()
        logging.getLogger(__name__).info(f"Retrieved image {photo.image_id} from GridFS")
        response = HttpResponse(image_data, content_type="image/jpeg")
        response['Content-Disposition'] = f'inline; filename="{photo.image_id}.jpg"'
        return response

    except Exception as e:
        return HttpResponse(f"Error retrieving image: {str(e)}", status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_similar_faces(request):
    photo_check = request.GET.get('type', '').lower() == 'face'
    if not photo_check:
        return JsonResponse({'error': 'Invalid photo type'}, status=400)
    photo_class = get_photo_class('face')
