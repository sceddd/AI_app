import datetime
import logging

from django.conf import settings
from djongo import models
import django.utils.timezone as timezone

mongo_client = settings.MONGO_CLIENT
logging.getLogger(__name__).setLevel(logging.DEBUG)
redis_client = settings.REDIS_CLIENT


class BoundingBox:
    bounding_boxes = models.JSONField(default=list)
    confidence = models.FloatField(default=0.0)
    class_name = models.CharField(max_length=30)

    def __init__(self, bboxes, conf, cls):
        self.bounding_boxes = bboxes
        self.confidence = conf
        self.class_name = cls

    def to_dict(self):
        return {
            'bboxes': self.bounding_boxes,
            'conf': self.confidence,
            'cls': self.class_name,
        }

    def __getitem__(self, name):
        return getattr(self, name)


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
    created_at = models.DateTimeField(default=timezone.now)
    is_new = models.BooleanField(default=True)
    bounding_boxes = models.JSONField(default=list)

    def add_bounding_box(self, bounding_box):
        self.bounding_boxes.append(bounding_box.to_dict())

    def __getitem__(self, name):
        return getattr(self, name)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.default_collection:
            raise ValueError('default_collection must be set')

    @classmethod
    def get_all_photos(cls):
        return cls.objects.all()[:100]

    def to_dict(self):
        return {
            'status': self.status,
            'image_id': self.image_id,
            'created_at': self.created_at.isoformat(),
            'bounding_boxes': [bbox for bbox in self.bounding_boxes],
        }

    def save(self, *args, **kwargs):
        if not self.is_new and self.pk is not None:
            logging.info(self.pk)
            original = type(self).objects.get(pk=self.pk)
            if original.is_new:
                try:
                    pass  # Your logic here
                except Exception as e:
                    logging.warning(f"DELFailed_Redis: {self.image_id} \t ex: {e}")
        super().save(*args, **kwargs)

    def switch_status(self):
        self.is_new = not self.is_new
        return self


class FaceEmbedding(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    face_id = models.CharField(max_length=30, primary_key=True)
    embedding = models.JSONField()

    def __getitem__(self, name):
        return getattr(self, name)

    def to_dict(self):
        return {
            'face_id': self.face_id,
            'embedding': self.embedding,
        }


class FacePhoto(AbstractPhoto):
    faces = models.JSONField(default=list)
    default_collection = 'face_images'

    def to_dict(self, get_faces_info=True, without_faces=False):
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
    default_collection = 'ocr_images'

    def to_dict(self):
        return {
            **super().to_dict(),
        }


class ObjectDetPhoto(AbstractPhoto):
    default_collection = 'od_images'

    def to_dict(self):
        return {
            **super().to_dict(),
        }


def get_photo_class(photo_type):
    if photo_type == 'face':
        return FacePhoto
    elif photo_type == 'ocr':
        return OCRPhoto
    elif photo_type == 'ob_det' or photo_type == 'det':
        return ObjectDetPhoto
    else:
        raise ValueError(f"Unknown photo type: {photo_type}")
