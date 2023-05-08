import os
import cv2
import torch
import sys
import numpy as np
import dataloader

from core.PaddleOCR.tools.infer import predict_rec
import core.PaddleOCR.tools.infer.utility as utility

cwd = os.getcwd()
PADDLEPATH = 'core/PaddleOCR'
sys.path.append(os.path.join(cwd, PADDLEPATH))

class ANPR:
    def __init__(self):
        #Initializing models
                        #path to cloned yolov5 dir
        self.model_lp = torch.hub.load('core/yolov5', 'custom', 
                            path='models/number_plate_det/number_plate_detection_yolov5s_best_198_16.pt',
                            device='0', source='local')
                        #path to cloned yolov5 dir
        self.model_gen_car = torch.hub.load('core/yolov5', 'custom', 
                            path='models/yolov5m_general/yolov5m.pt',
                            device='0', source='local')
                            #_verbose=False
        self.paddleocr_initialiser()

    def inference(self, im, crop=False):
        result_car = self.car_inference(im, crop)
        result_license = []
        if len(result_car)>0:
            for car in result_car:
                car_instance_im =im[int(car[1]):int(car[3]), int(car[0]):int(car[2])]
                res_lic=self.licence_inference(im=car_instance_im, crop=crop)
                #print(len(res_lic))
                if len(res_lic)>0:
                    y1, y2, x1, x2 = int(res_lic[0][1]), int(res_lic[0][3]), int(res_lic[0][0]), int(res_lic[0][2])
                    number_plate = self.paddleocr_inference([car_instance_im[y1:y2,x1:x2]])
                    #print(number_plate)
                else:
                    number_plate = "not_clear"
                
                result_license.append([[int(car[0]),int(car[1]),int(car[2]),int(car[3])],number_plate])
                #plot in im and write licence plate on car else not_found
            #print(result_license)
        else:
            result_license.append([[0,0,0,0],"not_clear"])
        return result_license
        
    def car_inference(self, im, crop):
        results = self.model_gen_car(im)
        else_crop_c = results.pandas().xyxy[0]
        else_crop_c = else_crop_c[else_crop_c['class'] == 2] #car is class 2
        else_crop_c = else_crop_c.values.tolist()
        # [xmin, ymin, xmax, ymax, confidence, class, name]
        #print(else_crop)
        return else_crop_c
    def licence_inference(self, im, crop):
        results = self.model_lp(im)
        if crop:
            crops = results.crop(save=True)
            #print(crops)
        else:
            else_crop_l = results.pandas().xyxy[0]
            else_crop_l = else_crop_l[else_crop_l['class'] == 0] #license plate is class 0
            else_crop_l = else_crop_l.values.tolist()
            # [xmin, ymin, xmax, ymax, confidence, class, name]
            #print(else_crop_l)
            return else_crop_l
        
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
        
    def paddleocr_initialiser(self):
        self.parser = utility.init_args()

        self.parser.add_argument('--image_dir', type=str, default="dataset/images/cars-1_license_plate.jpg")
        self.parser.add_argument('--rec_model_dir', type=str, default="models/paddle_ocr_recognition/ch_PP-OCRv3_rec_infer/")
        self.parser.add_argument('--use_gpu', type=self.str2bool, default=False)
        self.parser.add_argument('--rec_char_dict_path', type=str,
                            default="core/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt")
        self.args = self.parser.parse_args()
        self.parser.set_defaults(foo=None, )
        self.text_recognizer = predict_rec.TextRecognizer(self.args)

    def remove_argument(self, parser, arg):
        for action in parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                parser._remove_action(action)
                #print("removed")
                break

        for action in parser._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                #print("inside")
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return
    
    def paddleocr_inference(self, img):
        try:
            rec_res, _ = self.text_recognizer(img)
        except Exception as E:
            exit()
        return rec_res[0][0]


if __name__ == "__main__":
    anpr_1 = ANPR()
    dataset = dataloader.LoadImages("data/images/")
    writer = cv2.VideoWriter("output.mp4",
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    15, (1920, 1080))
    for path, im0s, vid_cap, s in dataset:
        print(path)
        resu = anpr_1.inference(im=im0s, crop=False)
        print(resu)
        #h,w,c = im0s.shape
        for r in resu:
            cv2.rectangle(im0s,(r[0][0],r[0][1]), (r[0][2],r[0][3]), (255, 0, 0), 2)
            cv2.putText(im0s,r[1], (r[0][0], r[0][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            #cv2.imshow("Show",im0s)
        writer.write(im0s)

    writer.release()
