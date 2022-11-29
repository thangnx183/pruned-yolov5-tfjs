import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

data_name = 'carpart-lr'
#modes = ['train','valid','test']
modes = ['filtered-train','filtered-valid']
json_path = Path('coco/carpart-lr/annotations_2022')
images_path = Path('coco/carpart-lr/images')
output_path = Path('coco_stuff_yolo')

for mode in modes :
    mode_path = output_path/data_name/mode
    mode_path.mkdir(parents=True,exist_ok=True)
    
    data = json.load(open(json_path/(mode+'.json')))
    cates = data['categories']
    
    for i in tqdm(data['images']):
        # image = cv2.imread(str(images_path/i['file_name']))
        # cv2.imwrite(str(mode_path/i['file_name']),image)
        width = i['width']
        height = i['height']

        with open(str(mode_path)+'.txt','a') as f:
            # f.write(str(mode_path/i['file_name'])+'\n') 
            f.write(str(images_path/i['file_name'])+'\n')

        note_text = i['file_name'][:i['file_name'].rfind('.')] + '.txt'
        annos = [a for a in data['annotations'] if a['image_id']==i['id']]

        for a in annos :
            bbox = a['bbox']
            x_c = str((bbox[0]+0.5*bbox[2])/width)
            y_c = str((bbox[1] + 0.5 * bbox[3]) / height)
            w = str(bbox[2] / width)
            h = str(bbox[3] / height)
            cate_id = str(a['category_id'])

            with open(mode_path/note_text,'a') as f:
                f.write(cate_id+' '+x_c+' '+y_c+' '+w+' '+h+'\n')
