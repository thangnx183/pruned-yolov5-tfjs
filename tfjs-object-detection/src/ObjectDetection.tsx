import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // Enable WebGL backend
// import { GraphModel } from "@tensorflow/tfjs";
import { DetectedObject, load, Yolo } from "./yolo";

const CARPART_NAMES = [
  "sli_side_turn_light",
  "tyre",
  "alloy_wheel",
  "hli_head_light",
  "hood",
  "fwi_windshield",
  "flp_front_license_plate",
  "door",
  "mirror",
  "handle",
  "qpa_quarter_panel",
  "fender",
  "grille",
  "fbu_front_bumper",
  "rocker_panel",
  "rbu_rear_bumper",
  "pillar",
  "roof",
  "blp_back_license_plate",
  "window",
  "rwi_rear_windshield",
  "tail_gate",
  "tli_tail_light",
  "fbe_fog_light_bezel",
  "fli_fog_light",
  "fuel_tank_door",
  "lli_low_bumper_tail_light",
  "car",
];

const CUSTOM_CONFIG = {
  source: "./best_web_prune_06_model_int8_320x320/model.json",
  classNames: CARPART_NAMES,
};

function ObjectDetection() {
  const [model, setModel] = useState<Yolo | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<DetectedObject[]>([]);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null); // State for inference time
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load the TensorFlow.js model when the component mounts
  useEffect(() => {
    load(CUSTOM_CONFIG)
      .then((loadedModel) => {
        setModel(loadedModel);
      })
      .catch((error) => {
        console.log("Error loading model:", error);
      });
  }, []);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setImageURL(imageUrl);
      setPredictions([]); // Clear previous predictions
      setInferenceTime(null); // Reset inference time
    }
  };

  // Function to run object detection on the uploaded image
  const detectObjects = async () => {
    if (!model || !imageRef.current) {
      console.log("Model not loaded or no image selected.");
      return;
    }

    const img = imageRef.current;
    const canvas = canvasRef.current;

    if (!canvas) {
      console.error("Canvas element is not available");
      return;
    }

    canvas.width = img.width;
    canvas.height = img.height;

    const ctx = canvas?.getContext("2d");

    if (!ctx) {
      console.error("Canvas context not available");
      return;
    }

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);

    // Measure inference time
    const startTime = performance.now(); // Start timer
    const modelOutput = await model.detect(img, 0.3); // Run model inference
    const endTime = performance.now(); // End timer

    // Calculate and set inference time
    const timeTaken = endTime - startTime;
    setInferenceTime(timeTaken); // Update state with inference time

    // Process model output
    setPredictions(modelOutput);
    drawBoundingBoxes(ctx, modelOutput, img.width, img.height);
  };

  const drawBoundingBoxes = (
    ctx: CanvasRenderingContext2D,
    objects: DetectedObject[],
    imgWidth: number,
    imgHeight: number
  ) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clear previous boxes
    if (imageRef.current) {
      // Check if imageRef.current is not null
      ctx.drawImage(imageRef.current, 0, 0, imgWidth, imgHeight); // Redraw the image
    } else {
      console.error("imageRef.current is null, cannot draw image");
    }

    // const labels = CUSTOM_CONFIG.classNames;

    objects.forEach((obj) => {
      const [x, y, width, height] = [obj.x, obj.y, obj.width, obj.height];

      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      const label = `Class ${obj.classId} (${(obj.score * 100).toFixed(2)}%)`;
      ctx.fillStyle = "red";
      ctx.font = "16px Arial";
      ctx.fillText(label, x, y > 10 ? y - 10 : y + 20); // Adjust label position if near top
    });
  };

  return (
    <div>
      <h1>Object Detection with TensorFlow.js</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {imageURL && (
        <div style={{ position: "relative" }}>
          <img
            src={imageURL}
            alt="Uploaded"
            ref={imageRef}
            style={{ maxWidth: "1500px", maxHeight: "1500px" }}
            onLoad={detectObjects} // Run detection when image loads
          />
          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              pointerEvents: "none",
            }}
            width={0}
            height={0}
          />
        </div>
      )}
      {predictions.length > 0 && (
        <div>
          <h2>Predictions:</h2>
          <ul>
            {predictions.map((prediction, index) => (
              <li key={index}>
                Class: {prediction.class}, Score:{" "}
                {(prediction.score * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
      {inferenceTime !== null && (
        <div>
          <h2>Inference Time: {inferenceTime.toFixed(2)} ms</h2>
        </div>
      )}
    </div>
  );
}

export default ObjectDetection;
