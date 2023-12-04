import './App.css';
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";
import * as bodySegmentation from "@tensorflow-models/body-segmentation";

function App() {
  const startImageScan = async () => {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    console.log("Starting");
    const element = document.getElementById("imageResult");
    element.innerHTML = 'Processing...';
    const image = document.getElementById("uploadedImage");

    const model = bodySegmentation.SupportedModels.BodyPix;

    const segmenterConfig = {
      runtime: "tfjs",
      modelType: "general",
      architecture: "ResNet50",
      outputStride: 16,
      quantBytes: 4
    };

    const segmenter = await bodySegmentation.createSegmenter(
      model,
      segmenterConfig
    );

    const segmentation = await segmenter.segmentPeople(image, {
      multiSegmentation: false,
      segmentBodyParts: true
    });

    const background = { r: 0, g: 0, b: 0, a: 0 };
    const foregroundThreshold = 0.7;

    const coloredPartImage = await bodySegmentation.toColoredMask(
      segmentation,
      bodySegmentation.bodyPixMaskValueToRainbowColor,
      background,
      foregroundThreshold
    );

    const opacity = 0.7;
    const flipHorizontal = false;
    const maskBlurAmount = 0;

    bodySegmentation.drawMask(
      canvas,
      image,
      coloredPartImage,
      opacity,
      maskBlurAmount,
      flipHorizontal
    );

    image.src = '';
    element.innerHTML = 'Check the result';
  };

  const onFileInput = () => {
    const selectedFile = document.getElementById("fileInput").files[0];

    const uploadedImage = document.getElementById('uploadedImage');
    uploadedImage.src=window.URL.createObjectURL(selectedFile); 
    startImageScan();
  }

  return (
    <>
      <h1>Recognize people with TensorFlow</h1>
      <div className='input'>
        <div className="imageResult" id="imageResult">
          Select file with people to recognize bodies
        </div>
        <div className="imageInput">
            <input type="file" id="fileInput" onInput={() => onFileInput()}/>
        </div>
        <img className="uploadedImage" id="uploadedImage" />
      </div>
      <canvas id='canvas'></canvas>
    </>
  )
}

export default App
