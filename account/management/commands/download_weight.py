from django.conf import settings
from django.core.management import BaseCommand
import os

from ._down_util import download_weights


class Command(BaseCommand):
    help = 'Download UMAP weights if not already downloaded.'

    def handle(self, *args, **kwargs):
        os.makedirs(settings.UMAP_DIR, exist_ok=True)

        weights = [
            ('UMAP', '17gcURWEy1BrQkk_zqWGeBIdyOS1Shz_0', settings.UMAP_WEIGHT_PATH),
            ('YOLO-WORLD',
             'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt',
             settings.YOLOW_WEIGHT_PATH),
            ('YOLOV8', '1-weMXsdqjylVh7G5c9KfLSGsyVQN4h67', settings.YOLOV8_WEIGHT_PATH),
            ('VGGFACE', 'https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth',settings.VGG),
        ]

        for name, url_or_id, path in weights:
            self.download_weight_if_needed(name, url_or_id, path)

    def download_weight_if_needed(self, name, url_or_id, path):
        if not os.path.exists(path):
            self.stdout.write(self.style.NOTICE(f'Downloading {name} weights to {path}...'))
            download_weights(url_or_id, path)
            self.stdout.write(self.style.SUCCESS(f'{name} weights downloaded successfully.'))
        else:
            self.stdout.write(self.style.SUCCESS('UMAP weights already exist.'))
