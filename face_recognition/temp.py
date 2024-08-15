import pickle

import lmdb

from face_recognition.face.det_model import convert_to_serializable
from face_recognition.face.util.inference import FaceRecognition

output_path = '/home/victor-ho/work/school/final/backend/myproject/lmdb/face/det'
lmdb_env = lmdb.open(output_path, readonly=True, lock=False)
face_idx = '66bc30aedebed778f8ec5e73'
results = {}
"""
VGGFace:
    weight_path: './vgg_face_dag.pth'
    img_size: 160,160
    function: .model.vgg, vgg_face_dag
    meta:
        imageSize: [224, 224, 3]
        mean: [129.186279296875, 104.76238250732422, 93.59396362304688]
        std: [1, 1, 1]
"""
cfg = {
    'VGGFace': {
        'weight_path': './face/weight/vgg_face_dag.pth',
        'img_size': (160, 160),
        'function': 'face.util.model.vgg, vgg_face_dag',
        'meta': {
            'imageSize': [224, 224, 3],
            'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
            'std': [1, 1, 1]
        }
    }
}
print(cfg['VGGFace'])
model = FaceRecognition(cfg['VGGFace'], 'cuda:0')
with lmdb_env.begin() as txn:
    stored_data = txn.get(face_idx.encode('utf-8'))
    if stored_data is None:
        print(f"No data found for image with idx {face_idx}")
        results[face_idx] = None

    face_data = pickle.loads(stored_data)

    new_pt = face_data.get("new_pt")

    embedding = model.embed(face_data.get("face"), new_pt)
    embedding = convert_to_serializable(embedding)
    results[face_idx] = embedding
print(results)
