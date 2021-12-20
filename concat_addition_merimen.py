from pathlib import Path 

path = Path('coco_stuff_yolo/carpart')
origin_path = path/'train_valid.txt'
merimen_path = path/'generalide.txt'

with open(origin_path) as f :
    origin_data = [line for line in f]

with open(merimen_path) as f :
    merimen_data = [line for line in f]

addition = [line for line in merimen_data if line not in origin_data]
print('duration :',len(merimen_data)-len(addition))
print('addition :',len(addition))
origin_data.extend(addition)

with open(path/'merge_train.txt','w') as f:
    for line in origin_data:
        f.write(line)
