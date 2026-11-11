"""Download official model dependencies after pip install -r requirements-study.txt."""
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import yaml
from detection_pipeline import ROOT,describe_device,resolve,sha256,write_json


def main():
    config=yaml.safe_load(resolve('config/detection_study.yaml').read_text())
    models=config['models']; actual=None
    # The YOLOX source and weights are only needed when the study lists a YOLOX model.
    yolox=[spec for spec in models.values() if spec['backend']=='yolox']
    if yolox:
        repository=resolve(config['yolox']['repository'])
        if not repository.exists():
            repository.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','https://github.com/Megvii-BaseDetection/YOLOX.git',str(repository)],check=True)
            subprocess.run(['git','-C',str(repository),'checkout',config['yolox']['commit']],check=True)
        actual=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
        if actual!=config['yolox']['commit']: raise ValueError('Existing YOLOX checkout differs; refusing to change it')
        for spec in yolox:
            weights=resolve(spec['weights'])
            if not weights.exists():
                weights.parent.mkdir(parents=True,exist_ok=True)
                temporary=weights.with_suffix('.download')
                urllib.request.urlretrieve(config['yolox']['weights_url'],temporary)
                temporary.replace(weights)
    from ultralytics import YOLO
    for spec in models.values():
        # The resolved path keeps the download beside the configuration, wherever this is run from.
        if spec['backend']=='ultralytics': YOLO(str(resolve(spec['weights'])))
    write_json(resolve('.ultralytics/study_dependencies.json'),{
        'yolox_commit':actual,'weights':{name:sha256(resolve(s['weights'])) for name,s in models.items()},
        'sources':{'yolox':config['yolox']['weights_url'] if yolox else None,'ultralytics':'official Ultralytics asset downloads'}})
    print('Official detector dependencies ready.')
    print(describe_device())


if __name__=='__main__': main()
