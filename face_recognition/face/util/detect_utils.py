import cv2
import numpy as np
from torchvision import transforms
from torch.nn.functional import interpolate
from .lfw_eval import get_similarity_transform_for_cv2
from .project_utils import cv2pillow


def process(data):
    img, landmark = data
    alg_img = alignment(img, landmark)
    pillow_fimg = cv2pillow(alg_img)
    return pillow_fimg, landmark


def cal_wh(box):
    """
    Calculate the width and height of a bounding box.

    Parameters:
    - box: List of bounding box coordinates [x_min, y_min, x_max, y_max]

    Returns:
    - (width, height): Tuple of width and height of the bounding box
    """
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return width, height


def calculate_flm_on_cropped_img(box, f_lm, imgcr_size=160):
    tl = box[:2]
    w, h = cal_wh(box)
    new_pt = []
    for p in f_lm:
        new_x = (p[0] - tl[0]) * (imgcr_size / w)
        new_y = (p[1] - tl[1]) * (imgcr_size / h)
        new_pt.append([new_x, new_y])
    return new_pt


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def compose_transforms(meta, resize=256, center_crop=True,
                       override_meta_imsize=False):
    """Compose preprocessing transforms for model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `resize`
           to select the image input size, rather than the properties contained
           in meta (this option only applies when center cropping is not used.

    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)


def alignment(src_img,src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def extract_features(net, ims):
    """Extract penultimate features from network

    Args:
        net (nn.Module): the network to be used to compute features
        ims (torch.Tensor): the data to be processed

    NOTE:
        Pretrained networks often vary in the manner in which their outputs
        are returned.  For example, some return the penultimate features as
        a second argument, while others need to be modified directly and will
        return these features as their only output.
    """
    outs = net(ims)
    if isinstance(outs, list):
        outs = outs[1]
    features = outs.data
    return features
