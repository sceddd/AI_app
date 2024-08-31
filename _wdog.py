import time
from pathlib import Path

from django.utils.autoreload import restart_with_reloader
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

BASE_DIR = Path(__file__).resolve().parent.parent


class CustomHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"File modified: {event.src_path}")
            restart_with_reloader()


if __name__ == "__main__":
    event_handler = CustomHandler()
    observer = Observer()
    observer.schedule(event_handler, path=BASE_DIR, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
