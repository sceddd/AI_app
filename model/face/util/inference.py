import logging
import os.path

import torch
from facenet_pytorch import MTCNN
from torch.autograd import Variable

from .detect_utils import *
from .downloads import download_weights
from .project_utils import create_module

logging.getLogger(__name__)


class FaceRecognition:
    def __init__(self, cfg, device,dowload_weight=False):
        self.cfg = cfg
        self.device = device
        self.img_size = cfg['img_size']
        self.meta = cfg['meta']
        self.function = cfg['function']
        logging.info(f'Using {cfg["function"]} model for recognition')
        weight_path = cfg['weight_path']
        if dowload_weight:
            download_weights(id_or_url='https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth',
                             cached=weight_path)
        else:
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f'Weight file {weight_path} not found')
        state_dict = torch.load(weight_path, map_location=self.device,weights_only=True)
        self.model = create_module(self.cfg['function'])(cfg).to(device)
        self.model.load_state_dict(state_dict)

        self.model.fc8 = torch.nn.Sequential()
        self.model.eval()

    def embed(self, img, landmark):

        img_alg, landmark = process((np.array(img.permute(1, 2, 0)).astype(np.uint8), landmark))

        img_alg = compose_transforms(meta=self.meta, center_crop=False)(img_alg)

        ims = Variable(torch.unsqueeze(img_alg, dim=0)).cuda()
        features = extract_features(self.model, ims)

        return features

    def recognition(self, input_recog):
        img1, img2 = input_recog['img1'][0], input_recog['img2'][0]

        features1 = self.embed(img1[0], img1[1])
        features2 = self.embed(img2[0], img2[1])

        f1, f2 = features1.squeeze(), features2.squeeze()
        cos = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        return cos


if __name__ == '__main__':
    model = MTCNN(
        image_size=160 ** 2, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device='cuda:0', keep_all=True
    )
    img = cv2.imread('/face_test/abc/0a0d7a87378422z3.jpg')
    torch.cuda.empty_cache()
    boxes, _ = model.detect(img)
    cv2.imwrite('output.jpg', img)
