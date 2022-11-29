from yolort.models import YOLOv5
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from yolort.utils import get_image_from_url, read_image_to_tensor
import torch
import onnx
import onnxsim
import onnxruntime
import cv2
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# CLASSES=  ['sli_side_turn_light', 'tyre', 'alloy_wheel', 'hli_head_light', 'hood',
#     'fwi_windshield', 'flp_front_license_plate', 'door', 'mirror', 'handle',
#     'qpa_quarter_panel', 'fender', 'grille', 'fbu_front_bumper', 'rocker_panel', 'rbu_rear_bumper',
#     'pillar', 'roof', 'blp_back_license_plate', 'window', 'rwi_rear_windshield',
#     'tail_gate', 'tli_tail_light', 'fbe_fog_light_bezel', 'fli_fog_light', 'fuel_tank_door',
#     'lli_low_bumper_tail_light'] 
CLASSES = ['r_sli_side_turn_light', 'l_sli_side_turn_light', 'r_tyre', 'l_tyre', 'r_alloy_wheel', 'l_alloy_wheel', 'r_hli_head_light', 'l_hli_head_light', 'hood', 'fwi_windshield', 'flp_front_license_plate', 'r_door', 'l_door', 'r_mirror', 'l_mirror', 'handle', 'r_qpa_quarter_panel', 'l_qpa_quarter_panel', 'r_fender', 'l_fender', 'grille', 'fbu_front_bumper', 'r_rocker_panel', 'l_rocker_panel', 'rbu_rear_bumper', 'r_pillar', 'l_pillar', 'roof', 'blp_back_license_plate', 'r_window', 'l_window', 'rwi_rear_windshield', 'tail_gate', 'r_tli_tail_light', 'l_tli_tail_light', 'r_fbe_fog_light_bezel', 'l_fbe_fog_light_bezel', 'r_fli_fog_light', 'l_fli_fog_light', 'fuel_tank_door', 'lli_low_bumper_tail_light', 'exhaust']

device = torch.device('cpu')
# 'yolov5s.pt' is downloaded from https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
ckpt_path_from_ultralytics = "runs/train/carpart_js2/weights/best.pt"
model = YOLOv5.load_from_yolov5(ckpt_path_from_ultralytics, score_thresh=0.4,nms_thresh=0.3)

debug_torch = False

if debug_torch:
    model.eval()
    print("+"*10)
    img_path = "coco/carpart/images/https:__s3.amazonaws.com_mc-ai_dataset_india_20190409_imgs_99_DSC06865.JPG"
    predictions = model.predict(img_path)
    print(predictions[0]['labels'].shape)
    img = cv2.imread(img_path)

    boxes = predictions[0]['boxes'].numpy().astype(np.int32)
    classes = predictions[0]['labels'].numpy().astype(np.int32)

    for i,box in enumerate(boxes) :
        box_class = CLASSES[classes[i]]
        img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,255,0),2)
        img = cv2.putText(img,str(box_class),(box[0],box[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)

    cv2.imwrite('torch_debug.jpg',img)

#img_one = get_image_from_url("https://gitee.com/zhiqwang/yolov5-rt-stack/raw/master/test/assets/bus.jpg")
img = cv2.imread('coco/carpart/images/https:__s3.amazonaws.com_mc-ai_dataset_india_20190409_imgs_99_DSC06865.JPG')
#img = cv2.imread('2178054924_423324aac8.jpg')
#img = cv2.resize(img,(416,416))
#print('shape : ',img.shape)
img_one = read_image_to_tensor(img, is_half=False)
print('image shape : ',img_one.shape)
img_one = img_one.to(device)

images = [img_one]

onnx_export = True

if onnx_export :
    export_onnx_name = 'carpart-tfjs.onnx'

    torch.onnx.export(
        model,
        (images,),
        export_onnx_name,
        do_constant_folding=True,
        opset_version=_onnx_opset_version, 
        input_names=["images_tensors"],
        output_names=["scores", "labels", "boxes"],
        dynamic_axes={
            "images_tensors": [0, 1, 2],
            "boxes": [0, 1],
            "labels": [0],
            "scores": [0],
        },
    )

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

onnx_debug = True

# export_onnx_name = 'runs/train/carpart_js2/weights/best.onnx'

if onnx_debug:
    inputs = list(map(to_numpy, images))

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if True else ['CPUExecutionProvider']

    print('initing ...')
    ort_session = onnxruntime.InferenceSession(export_onnx_name,providers=providers)
    ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
    print('inference ...')
    ort_outs = ort_session.run(['scores','labels','boxes'], ort_inputs)
    print('done')
    print('output label : ',ort_outs[1])



    for i,box in enumerate(ort_outs[2]):
        #print(box)
        box = np.array(box).astype(np.int32)
        print(box,ort_outs[1][i])
        box_class = CLASSES[ort_outs[1][i]]
        score = round(ort_outs[0][i],2)
        img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,255,0),2)
        img = cv2.putText(img,str(box_class)+'|'+str(score),(box[0],box[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
    #img = cv2.rectangle(img,(61,224),(350,415),(255,0,0),2)
    #img = cv2.rectangle(img,(140,66),(356,345),(255,0,0),2)
    cv2.imwrite('onnx_debug_cp.jpg',img)






























#print('simplifying ..')

# onnx_simp_name = 'demo.simp.onnx'

# onnx_model = onnx.load(export_onnx_name)

# # convert model
# model_simp, check = onnxsim.simplify(
#     onnx_model,
#     input_shapes={"images_tensors": [3, 640, 640]},
#     dynamic_input_shape=True,
# )

# assert check, "Simplified ONNX model could not be validated"

# #use model_simp as a standard ONNX model object
# onnx.save(model_simp, onnx_simp_name)

# 61.489654541015625
#  224.99612426757812
#  350.076171875
#  415.4963684082031
# 139.50808715820312
# 66.64620971679688
# 356.6035461425781
# 344.9726867675781