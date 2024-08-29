import os
import warnings
from django.core.management.commands.runserver import Command as RunserverCommand
from django.conf import settings


class Command(RunserverCommand):
    def handle(self, *args, **options):

        if not os.path.exists(settings.UMAP_WEIGHT_PATH):
            warnings.warn(f"UMAP weights not found at {settings.UMAP_WEIGHT_PATH}."
                     f"Please download them before running the server.")
            return

        super().handle(*args, **options)