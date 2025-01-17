import os
import urllib.request
import zipfile
import tarfile


data_dir = "./data/"
weights_dir = "./weights/"


def configure_paths_if_needed():
    # フォルダ「data」が存在しない場合は作成する    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    # フォルダ「weights」が存在しない場合は作成する    
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    
def download_datasets():
    # VOC2012のデータセットをここからダウンロードします
    # 時間がかかります（約15分）
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar") 

    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)
        
        tar = tarfile.TarFile(target_path)  # tarファイルを読み込み
        tar.extractall(data_dir)  # tarを解凍
        tar.close()  # tarファイルをクローズ
