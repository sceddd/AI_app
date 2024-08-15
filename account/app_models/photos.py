import base64
import json
from random import choices

from bson import ObjectId
from django.conf import settings
from djongo import models
from gridfs import GridFS

mongo_client = settings.MONGO_CLIENT
db = mongo_client['users_images']
fs = GridFS(db)


class FaceEmbedding(models.Model):
    face_id = models.CharField(max_length=30,primary_key=True)
    embedding = models.JSONField()
    gridfs_id = models.CharField(max_length=24, null=True, blank=True)

    def set_embedding(self, embedding):
        if isinstance(embedding, list):
            self.embedding = embedding
        else:
            raise ValueError('Embedding must be a float list')

    def get_embedding(self):
        return self.embedding

    def save_image(self, image):
        if isinstance(image, bytes):
            # image = base64.b64encode(image).decode('utf-8')
            self.gridfs_id = fs.put(image, filename=self.face_id)

    def get_image(self):
        return fs.get(self.gridfs_id).read()


class Photo(models.Model):
    class Status(models.IntegerChoices):
        UPLOADED = 1, 'Uploaded'
        PROCESSING = 2, 'Processing'
        EMBEDDING_SAVED = 3, 'Embedding Saved'
        COMPLETED = 4, 'Completed'
        ERROR = 5, 'Error'
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image_id = models.CharField(max_length=24, unique=True)
    status = models.IntegerField(choices=Status.choices, default=Status.UPLOADED)
    faces = models.ArrayReferenceField(to=FaceEmbedding,on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = models.Manager()

    def __str__(self):
        return self.image_id

    def save_image_to_gridfs(self, image_file):
        file_id = fs.put(image_file, filename=self.image_id)
        self.gridfs_id = str(file_id)
        self.save()

    def get_image_from_gridfs(self):
        return fs.get(ObjectId(self.gridfs_id)).read()
