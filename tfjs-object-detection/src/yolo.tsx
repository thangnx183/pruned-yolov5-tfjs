import * as tf from "@tensorflow/tfjs";
import { GraphModel } from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // Enable WebGL backend
import { create, all } from "mathjs";
const math = create(all);

interface DetectedObject {
  bbox?: number[];
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  classId: number;
  class: string;
}
class Yolo {
  tfModel: GraphModel; 
  modelWidth: number;
  modelHeight: number;
  categories: any;

  constructor(tfModel: GraphModel, categories: any) {
    this.tfModel = tfModel;

    // Initialize modelWidth and modelHeight with default values
    const shape = this.tfModel.inputs?.[0]?.shape;
    this.modelWidth = shape?.[1] ?? 320;
    this.modelHeight = shape?.[2] ?? 320;

    this.categories = categories;
  }

  preProcess(
    image: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData
  ) {
    let [processedImage, width, height, xRatio, yRatio] = tf.tidy(() => {
      let tensorImage = tf.browser.fromPixels(image);
      let [height, width] = tensorImage.shape.slice(0, 2);
      let maxSize = Math.max(height, width);
      let paddedImage = tensorImage.pad([
        [0, maxSize - height],
        [0, maxSize - width],
        [0, 0],
      ]);
      let processedImage2 = tf.image
        .resizeNearestNeighbor(paddedImage as tf.Tensor3D, [
          this.modelWidth,
          this.modelHeight,
        ])
        .div(255)
        .expandDims(0);
      return [
        processedImage2,
        width,
        height,
        maxSize / width,
        maxSize / height,
      ];
    });
    return [processedImage, width, height, xRatio, yRatio];
  }

  calculateMaxScores(
    scores: Float32Array,
    numBoxes: number,
    numClasses: number
  ): [number[], number[]] {
    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }

  buildDetectedObjects(
    width: number,
    xRatio: number,
    height: number,
    yRatio: number,
    boxes: Float32Array,
    scores: number[],
    indexes: Float32Array,
    classes: number[]
  ): DetectedObject[] {
    const count = indexes.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }

      const minY = math.max(0, bbox[0] * height * yRatio);
      const minX = math.max(0, bbox[1] * width * xRatio);
      const maxY = math.min(height, bbox[2] * height * yRatio);
      const maxX = math.min(width, bbox[3] * width * xRatio);

      objects.push({
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
        score: scores[indexes[i]],
        classId: classes[indexes[i]],
        class: this.categories[classes[indexes[i]]],
      });
    }
    return objects;
  }

  detect(
    image: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData,
    minScore = 0
  ) {
    let [processedImage, width, height, xRatio, yRatio] =
      this.preProcess(image);

    let modelOutput = this.tfModel.execute(processedImage as tf.Tensor);
    tf.dispose(processedImage);

    const output = modelOutput as tf.Tensor[]; // Ensure modelOutput is treated as a list of Tensor

    // Use dataSync to get the underlying data as Float32Array
    const boxesTensor = output[0]; // Get the boxes tensor
    const scoresTensor = output[1]; // Get the scores tensor

    const boxes = boxesTensor.dataSync() as Float32Array; // Convert boxes tensor to Float32Array
    const scores = scoresTensor.dataSync() as Float32Array; // Convert scores tensor to Float32Array
    
    // clean the webgl tensors
    if (processedImage instanceof tf.Tensor) {
      processedImage.dispose(); // Dispose only if it's a Tensor
    }

    const [maxScores, classes] = this.calculateMaxScores(
      scores,
      scoresTensor.shape[1] as number,
      scoresTensor.shape[2] as number
    );

    const prevBackend = tf.getBackend();
    // run post process in cpu
    if (tf.getBackend() === "webgl") {
      tf.setBackend("cpu");
    }

    const indexTensor = tf.tidy(() => {
      const boxes2 = tf.tensor2d(boxes, [
        boxesTensor.shape[1] as number,
        boxesTensor.shape[2] as number,
      ]);
      return tf.image.nonMaxSuppression(boxes2, maxScores, 100, 0.6, minScore);
    });
    tf.dispose(modelOutput);

    const indexes = indexTensor.dataSync() as Float32Array;
    indexTensor.dispose();

    // restore previous backend
    if (prevBackend !== tf.getBackend()) {
      tf.setBackend(prevBackend);
    }
    const detectionResults = this.buildDetectedObjects(
      width instanceof tf.Tensor ? width.dataSync()[0] : width,
      xRatio instanceof tf.Tensor ? xRatio.dataSync()[0] : xRatio,
      height instanceof tf.Tensor ? height.dataSync()[0] : height,
      yRatio instanceof tf.Tensor ? yRatio.dataSync()[0] : yRatio,
      boxes,
      maxScores,
      indexes,
      classes
    );
    tf.dispose(modelOutput);
    return detectionResults;
  }
}

// load model from local files
async function load(config: any) {
  const location = config.source;

  const modelUrl = location;

  // load model from url
  const tfModel = await tf.loadGraphModel(modelUrl);
  tf.enableProdMode();

  // warm-up inference
  const inputShape = tfModel.inputs[0].shape;

  if (inputShape) {
    tfModel.execute(tf.zeros(inputShape));
  } else {
    throw new Error("Input shape is undefined");
  }

  const model = new Yolo(tfModel, config.classNames);

  return model;
}

export { load, Yolo };

export type {
  // Change 'exports' to 'export type' and move DetectedObject here
  DetectedObject,
};
