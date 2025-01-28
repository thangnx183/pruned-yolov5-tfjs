# Redundant feature pruning with yolov5
This repo extend the work from paper 'Redundant Feature Pruning for Accelerated Inference in Deep Neural Networks' Babajide O. Ayindea, Tamer Inanca, Jacek M. Zuradaa.  
Contribution of this work:  
- apply proposed pruning method from paper for all yolov5 layer
- fine-tuned pruned model to regain accuracy (in my case, pruned half parameters of yolov5n without losing in accuracy)
- custom Tensorflowjs class to deploy on web: seprate nms module from model graph and execute in cpu for faster inference
- end2end pipeline from train - prune - finetune - convert and post training quantization tfjs - deploy tfjs

## Pruning 
```
python channel_prune.py --weight_path path/to/checkpoint --pruned_weight_path path/to/output/checkpoint --thres 0.5
```
- Specify model checkpoint, output of purned model in `weigth_path` and `pruned_weight_path`
- Specify cosin similarity distance threshold. The higher threshold the more filter weights are pruned

## Finetune to regain accuracy
```
python fine-tune.py --batch-size 144 --img 320 --data data/coco-old-cp.yaml --hyp data/hyps/hyp.scratch-high.yaml --weights runs/train/carpart-add-car-v2-anchor-tune-3-s-model-od/weights/best-prune-066.pt --sync-bn --device 2 --name prune-66-finetune-carpart-add-car-v2-anchor-tune-3-od-od
```
- Load pruned model checkpoint as pretrain and retrain model on same dataset

## Convert to TensorflowJs 
```
python export.py --weight path/to/pruned/checkpoint --int8 --includes tfjs
```

## Custom Yolov5js
- Move NMS separate from model graph, nms will excecute by cpu for faster inference ([detail](tfjs-object-detection/src/yolo.tsx))
- Link demo : [here](https://thangnx183.github.io/simple-tfjs-demo/)


