# file_watcher.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import faiss
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("users_data.csv"):
            print("Detected changes in user_data.csv. Updating vector store...")
            subprocess.run(["python", "preprocess.py", "data/usser_data.csv"])

if __name__ == "__main__":
    path = "data/"
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()