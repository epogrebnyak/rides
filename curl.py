import os
from pathlib import Path
from dataclasses import dataclass

import requests 
from tqdm import tqdm

from rides import Dataset

# warning: has problems with https on Windows
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

@dataclass
class Files:
    RAW_ZIP_FILE: str = datafile("jsons.zip")
    RAW_JSON_FOLDER: str = str (DATADIR / "jsons")
    TEMP_FOLDER: str = "temp"

class Getter:
    URL: str
    files: Files = Files() 
       
    def download(self):
        if not os.path.exists(self.files.RAW_ZIP_FILE):
            curl(self.URL, self.files.RAW_ZIP_FILE)
            
    def unzip(self):        
        import zipfile
        with zipfile.ZipFile(self.files.RAW_ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(self.files.RAW_JSON_FOLDER)
    
    @property
    def dataset(self):
        return Dataset(self.files.RAW_JSON_FOLDER, self.files.TEMP_FOLDER)
        
    def build(self):
        self.dataset.build()
    