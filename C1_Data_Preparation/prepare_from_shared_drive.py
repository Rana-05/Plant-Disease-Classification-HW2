"""Generate C1 dataset files from the original shared Google Drive folder.

Run in Colab after google.colab.auth.authenticate_user() and Drive mount.
Requires permission for original shared dataset and write access to OUTPUT_ROOT.
Downloads original images ONE TIME to a persistent folder, creates cleaned manifests.
Running again skips valid existing images; interrupted runs can be restarted.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from io import BytesIO
import hashlib, json, time, random
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

DATASET_FOLDER_ID = '1rkAM3SPGkMbStjyfk98fxIFOVB3EFZSK'
CLASS_NAMES = ['Apple___Apple_scab','Apple___Cedar_apple_rust','Apple___healthy',
               'Grape___Black_rot','Grape___healthy','Tomato___Bacterial_spot',
               'Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight']
LABELS = {name:i for i,name in enumerate(CLASS_NAMES)}
LAB_MAP = {
    ('Apple','Apple Scab'):CLASS_NAMES[0], ('Apple','Cedar Apple Rust'):CLASS_NAMES[1],
    ('Apple','Healthy'):CLASS_NAMES[2], ('Grape','Black Rot'):CLASS_NAMES[3],
    ('Grape','Healthy'):CLASS_NAMES[4], ('Tomato','Bacterial Spot'):CLASS_NAMES[5],
    ('Tomato','Early Blight'):CLASS_NAMES[6], ('Tomato','Healthy'):CLASS_NAMES[7],
    ('Tomato','Late Blight'):CLASS_NAMES[8],
}
FIELD_MAP = {('Apple',x):LAB_MAP[('Apple',x)] for x in ('Apple Scab','Cedar Apple Rust','Healthy')}
FIELD_MAP.update({('Grapes','Grape___Black_rot'):CLASS_NAMES[3],('Grapes','Grape___healthy'):CLASS_NAMES[4]})
FIELD_MAP.update({('Tomato',x):LAB_MAP[('Tomato',x)] for x in ('Bacterial Spot','Early Blight','Healthy','Late Blight')})
EXT = {'.jpg','.jpeg','.png','.webp','.bmp'}
_tls = local()
_creds, _ = google.auth.default()

def client():
    if not hasattr(_tls,'service'):
        _tls.service = build('drive','v3',credentials=_creds, cache_discovery=False)
    return _tls.service

def children(parent):
    token=None
    out=[]
    while True:
        r=client().files().list(q=f"'{parent}' in parents and trashed = false",fields='nextPageToken,files(id,name,mimeType)',pageSize=1000,pageToken=token,supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
        out += r.get('files',[])
        token = r.get('nextPageToken')
        if not token: return out

def child_folder(parent,name):
    matches=[x for x in children(parent) if x['name']==name and x['mimeType']=='application/vnd.google-apps.folder']
    if len(matches)!=1: raise ValueError(f'Expected one folder {name}, found {len(matches)}')
    return matches[0]['id']

def manifest(parent, crop_names, mapping, split):
    results=[]
    for crop in children(parent):
        if crop['name'] not in crop_names or crop['mimeType']!='application/vnd.google-apps.folder': continue
        for disease in children(crop['id']):
            if disease['mimeType']!='application/vnd.google-apps.folder': continue
            key=(crop['name'],disease['name'])
            if key not in mapping: raise ValueError(f'Unexpected class folder: {key}')
            label=mapping[key]
            for file in children(disease['id']):
                if Path(file['name']).suffix.lower() in EXT:
                    results.append({'file_id':file['id'],'filename':file['name'],'class_name':label,'label':LABELS[label],'split':split})
    return results

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def local_path(item, image_root):
    subfolder={'test':'plantdoc_test'}.get(item['split'],item['split'])
    return Path(image_root)/subfolder/item['class_name']/(item['file_id']+Path(item['filename']).suffix.lower())

def fetch(item, image_root):
    path=local_path(item,image_root)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size:
        try:
            with Image.open(path) as im: im.verify()
            return 'skipped'
        except Exception: path.unlink(missing_ok=True)
    partial=path.with_name(path.name+'.part')
    for attempt in range(5):
        try:
            req=client().files().get_media(fileId=item['file_id'],supportsAllDrives=True)
            with open(partial,'wb') as output:
                downloader=MediaIoBaseDownload(output,req)
                done=False
                while not done: _,done=downloader.next_chunk()
            with Image.open(partial) as im: im.verify()
            partial.replace(path)
            return 'downloaded'
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt<4: time.sleep(min(60,2**attempt+random.random()))
    return 'failed'

def prepare(output_root, max_workers=4):
    output_root=Path(output_root)
    image_root=output_root/'prepared_data'
    splits=output_root/'splits'; splits.mkdir(parents=True,exist_ok=True)
    config=output_root/'config'; config.mkdir(parents=True,exist_ok=True)
    source=child_folder(DATASET_FOLDER_ID,'Plant_Disease_Detection_Dataset')
    field=child_folder(DATASET_FOLDER_ID,'PlantDoc_field')
    # Use existing train/valid split; do not train on source/test.
    records=[]
    for name in ('train','valid'):
        records += manifest(child_folder(source,name),{'Apple','Grape','Tomato'},LAB_MAP,name)
    records += manifest(field,{'Apple','Grapes','Tomato'},FIELD_MAP,'test')
    if len({r['file_id'] for r in records})!=len(records): raise ValueError('Repeated Google Drive file ID in manifests')
    print('Source counts:',pd.Series([r['split'] for r in records]).value_counts().to_dict())
    # Save index even if a transfer gets interrupted; download is resumable.
    pd.DataFrame(records).to_csv(splits/'raw_index.csv',index=False)
    failed=[]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures={pool.submit(fetch,r,image_root):r for r in records}
        for fut in tqdm(as_completed(futures),total=len(futures),desc='Saving images'):
            if fut.result()=='failed': failed.append(futures[fut])
    if failed:
        pd.DataFrame(failed).to_csv(splits/'failed_downloads.csv',index=False)
        raise RuntimeError(f'{len(failed)} images failed. Rerun prepare() to retry; completed images will be skipped.')
    # Hash each verified file to remove duplicates and detect conflicts.
    grouped=defaultdict(list)
    for r in tqdm(records,desc='Hashing image files'):
        grouped[sha256(local_path(r,image_root))].append(r)
    conflicts=[group for group in grouped.values() if len({r['label'] for r in group})>1]
    if conflicts: raise ValueError(f'{len(conflicts)} hashes have conflicting labels; inspect before training')
    # Train takes precedence over valid; valid over test. All splits remain disjoint.
    priority={'train':0,'valid':1,'test':2}
    chosen=[]
    for group in grouped.values():
        chosen.append(sorted(group,key=lambda r:(priority[r['split']],r['file_id']))[0])
    result=pd.DataFrame(chosen)
    for name in ('train','valid','test'):
        subset=result[result['split']==name].drop(columns=['split']).sort_values(['label','file_id'])
        if subset.empty: raise ValueError(f'Cleaned {name} has no images')
        subset.to_csv(splits/f'{name}.csv',index=False)
        print(f'Clean {name}:',len(subset))
    counts=result[result['split']=='train']['label'].value_counts().reindex(range(9),fill_value=0)
    if (counts==0).any(): raise ValueError('Training class missing')
    import torch
    weights=torch.as_tensor(len(result[result['split']=='train'])/(9*counts.to_numpy()),dtype=torch.float32)
    torch.save(weights,config/'class_weights.pt')
    (config/'class_to_idx.json').write_text(json.dumps(LABELS,indent=2))
    (config/'project_config.json').write_text(json.dumps({'image_size':224,'batch_size':32,'seed':42,'test_dataset':'PlantDoc','models':['FasterNet','EfficientViT','GhostNet']},indent=2))
    print('READY. Verify Google Drive web UI contains files under',output_root)
    return result
