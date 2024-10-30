import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader
from ultocr.inference import Detection, Recognition
from PIL import Image

from ultocr.loader.recognition.reg_loader import TextInference
from ultocr.loader.recognition.reg_loader import Resize
from ultocr.model.recognition.postprocess import greedy_decode_with_probability

from ocr_util.ocr_util import sort_by_line, four_point_transform, draw_bbox, test_preprocess, read_config


class DBDet(Detection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, img):
        det_result = {}
        h_origin, w_origin = img.shape[:2]
        tmp_img = test_preprocess(img, new_size=736, pad=False).to(self.device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            preds = self.model(tmp_img)
        batch = {'shape': [(h_origin, w_origin)]}
        boxes_list, scores_list = self.seg_obj(batch, preds, inference=True)
        boxes_list, scores_list = boxes_list[0].tolist(), scores_list[0]

        boxes_list.sort(key=lambda x: x[0][1])
        boxes_list_remove = []
        for boxes in boxes_list:
            if boxes[0] == boxes[2] or boxes[1] == boxes[3]:
                continue
            else:
                boxes_list_remove.append(boxes)
        # No text detected
        if len(boxes_list_remove) == 0:
            det_result['img'] = img
            det_result['box_coordinate'] = []
            det_result['boundary_result'] = []
            return det_result
        else:
            sort_box_list = sort_by_line(boxes_list_remove)
            after_sort = []
            after_sort2 = []
            for same_row in sort_box_list:
                for box in same_row:
                    point = np.array(box)
                    point = point.astype(int)
                    after_sort2.append(point)
                    box = np.array(box).reshape(-1).tolist()
                    after_sort.append(box)

            rs = draw_bbox(img, np.array(after_sort), color=(0, 0, 255), thickness=2)
            all_warped = []
            for index, boxes in enumerate(after_sort2):
                warped = four_point_transform(img, boxes)
                all_warped.append(warped)

            det_result['img'] = rs
            det_result['box_coordinate'] = after_sort
            det_result['boundary_result'] = all_warped
            return det_result


class MasterReg(Recognition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def recognize(self, list_img):
        text_dataset = TextInference(list_img, transform=Resize(self.img_w, self.img_h, gray_format=False))
        text_loader = DataLoader(text_dataset, batch_size=self.batch, shuffle=False, num_workers=4, drop_last=False)
        pred_results = []
        for step_idx, data_item in enumerate(text_loader):
            images = data_item

            with torch.no_grad():
                images = images.to(self.device)
                outputs, probs = greedy_decode_with_probability(self.model, images, self.convert.max_length,
                                                                self.convert.SOS,
                                                                padding_symbol=self.convert.PAD,
                                                                device=self.device, padding=True)

            for index, (pred, prob) in enumerate(zip(outputs[:, 1:], probs)):
                pred_text = ''
                previous_text = None
                repeat = 0
                pred_score_list = []
                for i in range(len(pred)):
                    if pred[i] == self.convert.EOS:
                        pred_score_list.append(prob[i])
                        break
                    if pred[i] == self.convert.UNK:
                        continue

                    decoder_char = self.convert.decode(pred[i])
                    if decoder_char == previous_text:
                        repeat += 1
                    previous_text = decoder_char
                    if repeat == 5:
                        break
                    pred_text += decoder_char
                    pred_score_list.append(prob[i])
                pred_results.append(pred_text)
        return pred_results


class MyOCR:
    def __init__(self, det_model, reg_model, det_weight, det_cfg, reg_weight, reg_cfg, device):
        assert det_model in ['DB'], '{} model is not implement'.format(det_model)
        assert reg_model in ['MASTER'], '{} model is not implement'.format(reg_model)
        self.detection = DBDet(weight=det_weight, cfg=read_config(det_cfg), device=device)
        self.recognition = MasterReg(weight=reg_weight, cfg=read_config(reg_cfg), device=device)

    def get_result(self, image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det_result = self.detection.detect(img)
        all_img_crop = det_result['boundary_result']
        boxes_coordinate = det_result['box_coordinate']

        if len(all_img_crop) == 0:
            result = 'No text detected'
            return result
        all_img_pil = []

        for idx, img_crop in enumerate(all_img_crop):
            img_pil = Image.fromarray(img_crop.astype('uint8'), 'RGB')
            all_img_pil.append(img_pil)

        result = self.recognition.recognize(all_img_pil)
        infos = dict()
        infos['boxes'] = boxes_coordinate
        infos['texts'] = result
        infos['img_with_box'] = det_result['img']
        infos['crop_image'] = all_img_crop
        return infos


if __name__ == '__main__':
    model = MyOCR(det_model='DB', reg_model='MASTER',
                  det_cfg='./ocr_util/config/db.yaml',reg_cfg='./ocr_util/config/master.yaml'
                  ,det_weight='./weights/db.pth', reg_weight='./weights/master.pth',device='cpu')
    image = Image.open('test/input.png')
    result = model.get_result(image)
    # print(result['boxes'])
    # print(result['texts'])
    # for idx,img in enumerate(result['crop_image']):
    #     cv2.imwrite(f'test/crop_image{idx}.jpg', img)
    # cv2.imwrite('test/result_image.jpg', result['img_with_box'])
