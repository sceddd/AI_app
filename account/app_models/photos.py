from bson import ObjectId
from django.conf import settings
from djongo import models
from gridfs import GridFS
import logging
from AI_Backend.settings import IMAGE_DB

mongo_client = settings.MONGO_CLIENT
logging.getLogger(__name__).setLevel(logging.DEBUG)
redis_client = settings.REDIS_CLIENT


class AbstractPhoto(models.Model):
    class Status(models.IntegerChoices):
        UPLOADED = 1, 'Uploaded'
        PROCESSING = 2, 'Processing'
        RESULT_SAVED = 3, 'Result Saved'
        COMPLETED = 4, 'Completed'
        ERROR = 5, 'Error'

    class Meta:
        abstract = True

    default_collection = None
    image_id = models.CharField(max_length=24, unique=True, primary_key=True)
    status = models.IntegerField(choices=Status.choices, default=Status.UPLOADED)
    created_at = models.DateTimeField(auto_now_add=True)
    bounding_boxes = models.JSONField(default=list)
    gridfs_id = models.CharField(max_length=24, null=True, blank=True)
    is_new = models.BooleanField(default=True)

    def __getitem__(self, name):
        return getattr(self, name)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.default_collection:
            raise ValueError('default_collection must be set')

    @classmethod
    def get_all_photos(cls):
        return cls.objects.all()

    def to_dict(self):
        return {
            'status': self.status,
            'image_id': self.image_id,
            'created_at': self.created_at.isoformat(),  # Convert datetime to string in ISO format
            'bounding_boxes': self.bounding_boxes,
        }

    def save_image_to_gridfs(self, image_file):
        file_id = GridFS(IMAGE_DB, collection=self.default_collection).put(image_file, filename=self.image_id)
        self.gridfs_id = str(file_id)
        self.save(update_fields=['gridfs_id'])

    def get_image_from_gridfs(self):
        return GridFS(IMAGE_DB, collection=self.default_collection).get(ObjectId(self.gridfs_id)).read()

    def save(self, *args, **kwargs):
        if not self.is_new and self.pk is not None:
            original = type(self).objects.get(pk=self.pk)
            if original.is_new:
                try:
                    self.is_new = False
                    redis_client.delete(self.image_id)
                except Exception as e:
                    logging.warning(f"DELFailed_Redis: {self.image_id} \t ex: {e}")
        super().save(*args, **kwargs)

    def switch_status(self):
        self.is_new = not self.is_new
        return self


class FaceEmbedding(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = GridFS(IMAGE_DB, collection='face_images')

    face_id = models.CharField(max_length=30, primary_key=True)
    embedding = models.JSONField()
    gridfs_id = models.CharField(max_length=24, null=True, blank=True)

    def __getitem__(self, name):
        return getattr(self, name)

    def save_image(self, image):
        if isinstance(image, bytes):
            self.gridfs_id = self.db.put(image, filename=ObjectId(self.face_id))

    def get_image(self):
        return self.db.get(self.gridfs_id).read()

    def to_dict(self):
        return {
            'face_id': self.face_id,
            'embedding': self.embedding,
        }


class FacePhoto(AbstractPhoto):
    faces = models.JSONField(default=list)
    default_collection = 'face_images'

    def to_dict(self, get_faces_info=True):
        faces_info = []
        if get_faces_info:
            for face_id in self.faces:
                try:
                    face_embedding = FaceEmbedding.objects.get(face_id=face_id)
                    faces_info.append(face_embedding.to_dict())
                except FaceEmbedding.DoesNotExist:
                    faces_info.append({'face_id': face_id, 'error': 'FaceEmbedding not found'})
        return {
            **super().to_dict(),
            'faces': faces_info if get_faces_info else self.faces,
        }


class OCRPhoto(AbstractPhoto):
    texts = models.JSONField(default=list)
    default_collection = 'ocr_images'

    def to_dict(self):
        return {
            **super().to_dict(),
            'texts': self.texts,
        }


class ObjectDetPhoto(AbstractPhoto):
    objects_det = models.JSONField(default=list)
    default_collection = 'od_images'

    def to_dict(self):
        return {
            **super().to_dict(),
            'objects_det': self.objects_det,
        }


def get_photo_class(photo_type):
    if photo_type == 'face':
        return FacePhoto
    elif photo_type == 'ocr':
        return OCRPhoto
    elif photo_type == 'ob_det':
        return ObjectDetPhoto
    else:
        raise ValueError(f"Unknown photo type: {photo_type}")
