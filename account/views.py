import logging
import uuid

import lmdb

from celery.result import AsyncResult
from django.conf import settings
from django.contrib.auth import login, authenticate, get_user_model, logout
from django.db import DatabaseError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import permission_classes, api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken

from .forms.login_form import CustomAuthenticationForm
from .forms.register_form import CustomUserCreationForm
from .models import CustomUser
from .project_utils.utils import get_photo
from .tasks.tasks import find_similar, process_zip_file_lmdb, process_ob_det

User = get_user_model()
logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def user_photos(request):
    print('GET: account/user_photos')
    if request.method == 'GET':
        photo_type = request.GET.get('type','').lower()
        user = request.user
        photos_id = user.get_image(photo_type)
        photo_class = get_photo(photo_type)
        photos = photo_class.objects.filter(image_id__in=photos_id)
        result = []
        for photo in photos:
            try:
                result.append(photo.to_dict())
            except Exception as e:
                logger.error(f"Error retrieving photo {photo.image_id}, error: {e}")
        return JsonResponse({'photos':result}, status=200)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_task_status(request, task_id):
    print('GET: account/task_status')
    print('received task_id:', task_id)
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return JsonResponse({'state': task_result.state}, status=202)
    elif task_result.state == 'SUCCESS':
        return JsonResponse({'state': task_result.state, 'result': task_result.result}, status=200)
    elif task_result.state == 'FAILURE':
        return JsonResponse({'state': task_result.state, 'error': str(task_result.info)}, status=500)
    else:
        return JsonResponse({'state': task_result.state}, status=202)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def get_similar_faces(request):
    print('POST: account/get_similar_faces')
    photo_check = request.POST.get('type', '').lower() == 'face'
    if not photo_check:
        return JsonResponse({'error': 'Invalid photo type'}, status=400)
    user = request.user

    k_faces = request.POST.get('k', 4)
    photo_id = request.POST.get('photo_id', '')

    if k_faces <= 0:
        return JsonResponse({'error': 'Invalid value for k'}, status=400)
    if not photo_id:
        return JsonResponse({'error': 'photo_id is required'}, status=400)
    response = []
    list_photos = set(user.get_image('face'))
    if photo_id in list_photos:
        list_photos.remove(photo_id)
    photo_class = get_photo('face')
    photos = photo_class.objects.filter(image_id__in=list_photos)
    faces_dict = {photo.image_id: photo.to_dict().get('faces') for photo in photos if photo.faces}
    try:
        photo_faces = get_object_or_404(photo_class,image_id=photo_id)
        if photo_faces.to_dict().get('status') != 3:
            return JsonResponse({'error': 'Photo not processed yet'}, status=400)

        result = photo_faces.to_dict()
        for idx, face in enumerate(result['faces']):
            if 'embedding' not in face:
                continue
            temp = f"0,{idx},1,{photo_id}"
            result['output'] = [temp]
            similar_faces = find_similar(faces_dict, face, k_faces)
            f_id = 1
            for face_data in similar_faces['k_faces']:
                img_id = face_data['image_id']
                distance = face_data['distance']
                similar_photo = photo_class.objects.filter(image_id=img_id)
                result['bounding_boxes'].extend(similar_photo[0].to_dict().get('bounding_boxes'))
                temp = f"{f_id},{distance:.2f},{img_id}"
                f_id += 1
                result['output'].append(temp)
        response.append(result)
    except photo_class.DoesNotExist:
        return JsonResponse({'error': 'Photo not found'}, status=404)
    except IndexError:
        return JsonResponse({'error': 'No face detected in this photo'}, status=400)
    return JsonResponse({'results':result}, status=202)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def predict_objects(request):
    print('POST: account/predict_objects')
    photo_check = request.POST.get('type', '').lower() == 'ob_det'
    if not photo_check:
        return JsonResponse({'error': 'Invalid photo type'}, status=400)
    user = request.user
    photo_id = request.POST.get('photo_id', '')
    photo_class = get_photo('ob_det')
    input_words = request.POST.get('input_words', '')

    if not input_words:
        return JsonResponse({'error': 'input_words is required'}, status=400)
    try:
        photo = get_object_or_404(photo_class, image_id=photo_id)
        task = process_ob_det.apply_async(args=[photo_id,input_words],queue="image_processing")
    except photo_class.DoesNotExist:
        return JsonResponse({'error': 'Photo not found'}, status=404)
    return JsonResponse({'task_id': task.id}, status=202)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def upload(request):
    print('POST: account/upload')
    if request.method == 'POST' and 'file' in request.FILES:
        user = request.user
        file = request.FILES.get('file')
        photo_type = request.POST.get('type', '').lower()
        if not photo_type or photo_type not in ['face', 'ocr', 'ob_det']:
            return JsonResponse({'error': 'Invalid photo type'}, status=400)

        if not file.name.endswith('.zip'):
            return JsonResponse({'error': 'Only .zip files are allowed'}, status=400)

        zip_key = f'zip_{uuid.uuid4()}_t_{photo_type}'

        lmdb_env = lmdb.open(str(settings.ZIP_PATH), map_size=settings.LMDB_LIMIT * settings.LMDB_BATCH_SIZE)

        with lmdb_env.begin(write=True) as txn:
            zip_data = b''.join(chunk for chunk in file.chunks())
            txn.put(zip_key.encode('utf-8'), zip_data)

        task = process_zip_file_lmdb.apply_async(args=[zip_key, user.id],queue="image_upload")
        print(f"Task ID: {task.id}")
        return JsonResponse({'task_id': task.id}, status=202)

    return JsonResponse({'error': 'No zip file found'}, status=400)


@csrf_exempt
def login_user(request):
    print('POST: account/login')
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                tokens = get_tokens_for_user(user)

                # Retrieve the user id (which is the ObjectId in MongoDB)
                user_id = user.pk  # Access the MongoDB _id via the Django 'pk' attribute

                return JsonResponse({
                    "message": "User logged in successfully",
                    "tokens": tokens,
                    "user_id": user_id  # Return the ObjectId (user id) here
                }, status=200)
            else:
                return JsonResponse({"error": "Invalid password"}, status=400)
        else:
            return JsonResponse({"errors": form.errors}, status=400)

    return JsonResponse({"error": "Only POST method is allowed"}, status=405)


@csrf_exempt
def register_user(request):
    print('POST: account/register')
    if request.method == 'POST':
        try:
            form = CustomUserCreationForm(request.POST)
            if form.is_valid():
                user = form.save()
                return JsonResponse({"message": "User created successfully", "token":
                                     get_tokens_for_user(user)}, status=201)
            else:
                return JsonResponse({"errors": form.errors}, status=400)
        except DatabaseError as e:
            return JsonResponse({'error': 'Databases error: '+str(e)}, status=404)
    return JsonResponse({"error": "Only POST method is allowed"}, status=405)


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_image(request):
    try:
        print('GET: account/get_image')
        photo_type = get_photo(request.GET.get('type', '').lower())
        image_id = request.GET.get('id', '')
        photo = get_object_or_404(photo_type, image_id=image_id)
        image_data = photo.to_dict()
        return JsonResponse(image_data)

    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def logout_user(request):
    if request.method == 'POST':
        logout(request)
        return JsonResponse({"message": "User logged out successfully"}, status=200)
    return JsonResponse({"error": "Only POST method is allowed"}, status=405)


@api_view(['POST'])
@csrf_exempt
def refresh_access_token(request):
    if request.method == 'POST':
        refresh_token = request.data.get('refresh', '')
        if not refresh_token:
            return JsonResponse({'error': 'Refresh token is required'}, status=400)

        try:
            refresh = RefreshToken(refresh_token)

            user_id = refresh['user_id']
            user = CustomUser.objects.get(id=user_id)

            if not user.is_active:
                return JsonResponse({'error': 'User is inactive or deleted'}, status=403)

            access_token = str(refresh.access_token)
            return JsonResponse({
                'access': access_token,
                'refresh': str(refresh)
            }, status=200)
        except CustomUser.DoesNotExist:
            return JsonResponse({'error': 'User not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
