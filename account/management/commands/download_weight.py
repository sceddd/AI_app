from django.conf import settings
from django.core.management import BaseCommand
import os
from ._down_util import download_weights

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Command(BaseCommand):
    help = 'Download UMAP weights if not already downloaded.'

    def handle(self, *args, **kwargs):
        self.stdout.write("This is a test command.")
        os.makedirs(settings.UMAP_DIR, exist_ok=True)
        UMAP_WEIGHT_PATH = settings.UMAP_WEIGHT_PATH
        if not os.path.exists(UMAP_WEIGHT_PATH):
            self.stdout.write(self.style.NOTICE(f'Downloading weights to {UMAP_WEIGHT_PATH}...'))
            download_weights('17gcURWEy1BrQkk_zqWGeBIdyOS1Shz_0',
                             UMAP_WEIGHT_PATH)
            self.stdout.write(self.style.SUCCESS('UMAP weights downloaded successfully.'))
        else:
            self.stdout.write(self.style.SUCCESS('UMAP weights already exist.'))
