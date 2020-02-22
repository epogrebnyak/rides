from pathlib import Path
import requests 
from tqdm import tqdm
import os

from rides import Dataset


URL = ("https://dl.dropboxusercontent.com/" 
       "sh/hibzl6fkzukltk9/AABTFmhvDvxyQdUaBsKl4h59a/"
       "data_samples_json.zip")

def curl(path: str, url: str, stop_at=None):
    r = requests.get(url, stream=True)
    iterable = tqdm(r.iter_content(chunk_size=1024), unit=' k')
    with open(path, 'wb') as f:
        i = 0 
        for chunk in iterable:
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                i+=1
                if stop_at and i > stop_at:
                    break
                
DATADIR = Path("data")  
DATADIR.mkdir(parents=False, exist_ok=True)

def datafile(filename):
    return str(DATADIR / filename)

class Files:  
    RAW_ZIP_FILE = datafile("jsons.zip")
    RAW_JSON_FOLDER = str (DATADIR / "jsons")
    TEMP_FOLDER = "temp"
    
    def __init__(self, url):
        self.url = url
        
    def download(self):
        if not os.path.exists(self.RAW_ZIP_FILE):
            curl(self.url, self.RAW_ZIP_FILE)
            
    def unzip(self):        
        import zipfile
        with zipfile.ZipFile(self.RAW_ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(self.RAW_JSON_FOLDER)
    
    @property
    def dataset(self):
        return Dataset(self.RAW_JSON_FOLDER, self.TEMP_FOLDER)
        
    def build(self):
        self.dataset.build()
    