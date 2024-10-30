import logging

import torch
from torch.autograd import Variable

from .detect_utils import *
from .project_utils import create_module, cv2pillow, read_config

logging.getLogger(__name__)


class FaceRecognition:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.meta = cfg['meta']
        self.function = cfg['function']
        weight_path = cfg['weight_path']
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.alg_params = cfg['alg_params']
        self.model = create_module(self.cfg['function'])(cfg).to(device)
        self.model.load_state_dict(state_dict)
        self.model.fc8 = torch.nn.Sequential()
        self.model.eval()

    def embed(self, img, landmark):

        def process(data):
            img, landmark = data
            alg_img = alignment(img, landmark,**self.alg_params)
            pillow_fimg = cv2pillow(alg_img)
            return pillow_fimg, landmark

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
    cfg = read_config('./configs/recognition_model.yaml')
    model = FaceRecognition(cfg['VGGFace'], torch.device('cuda'))
    model.embed(torch.randn(3, 112, 112), torch.randn(5, 2))
