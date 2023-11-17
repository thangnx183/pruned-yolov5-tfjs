# python export.py --weights runs/train/carpart-add-car-v2-anchor-tune/weights/best.pt --int8 --data data/coco-old-cp.yaml --imgsz 480 --include tflite
# python val.py --data data/coco-old-cp.yaml --weights runs/train/carpart-add-car-v2-anchor-tune/weights/best.pt --batch-size 8 --device 0 --imgsz 480

python export.py --weights runs/train/carpart-add-car-v2-anchor-tune-3-gen2/weights/best.pt --int8 --data data/coco-old-cp.yaml --imgsz 480 --include tfjs
# du -sh prune-finetune-carpart-add-car-75-no-mosaic/weights/*web_model*