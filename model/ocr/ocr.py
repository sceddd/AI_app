import cv2
import numpy as np
from ultocr.inference import OCR
from PIL import Image


class MyOCR(OCR):
    def __init__(self, det_model='DB', reg_model='MASTER'):
        super().__init__(det_model, reg_model)

    def get_result(self, image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det_result = self.detection.detect(img)
        all_img_crop = det_result['boundary_result']
        boxes_coordinate = det_result['box_coordinate']
        if len(all_img_crop) == 0:
            result = 'khong co text trong anh'
            return result
        all_img_pil = []
        for idx, img_crop in enumerate(all_img_crop):
            img_pil = Image.fromarray(img_crop.astype('uint8'), 'RGB')
            all_img_pil.append(img_pil)
        result = self.recognition.recognize(all_img_pil)
        infos = dict()
        infos['boxes'] = boxes_coordinate
        infos['texts'] = result
        return infos


if __name__ == '__main__':
    model = MyOCR(det_model='DB', reg_model='MASTER')

    image = Image.open('/home/victor-ho/work/school/final/TextDetection/model/DBNet_Reg/datasets/train_data/train_imgs/img_1.jpg')  # ..is the path of image
    result,img_wb,ls_crop = model.get_result(image)
    print(result)
    print(result['texts'])
    Image.fromarray(img_wb).show('detected_image')
    num_images = 5
    print(len(ls_crop))
