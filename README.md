# Redundant feature pruning with yolov5
This repo extend the work from paper 'Redundant Feature Pruning for Accelerated Inference in Deep Neural Networks' Babajide O. Ayindea, Tamer Inanca, Jacek M. Zuradaa.  
Contribution of this work:  
- apply proposed pruning method from paper for all yolov5 layer
- fine-tuned pruned model to regain accuracy
- custom Tensorflowjs class to deploy on web 

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
