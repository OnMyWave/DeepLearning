import os 
import tarfile
import urllib

DOWNLOAD_ROOT = "https://"
PATH = os.path.join('datasets')
URL = DOWNLOAD_ROOT + ''

def fetch_data(url = URL, path = PATH):
    os.makedirs(path, exist_ok= True)
    tgz_path = os.path.join(path, 'file_name')
    urllib.request.urlretrieve(url,tgz_path)
    tgz = tarfile.open(tgz_path)
    tgz.extractfile(path = path)
    tgz.close()
