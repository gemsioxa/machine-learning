import './App.css';
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";
import * as bodySegmentation from "@tensorflow-models/body-segmentation";

function App() {

  const dropHandler = (e) => {
    console.log(e);
  };

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

    const foreground = { r: 0, g: 0, b: 0, a: 0 };
    const background = { r: 0, g: 0, b: 0, a: 255 };
    const drawContour = true;
    const foregroundThreshold = 0.6;

    const binaryPartImage = await bodySegmentation.toBinaryMask(
      segmentation,
      // bodySegmentation.bodyPixMaskValueToRainbowColor,
      foreground,
      background,
      drawContour,
      foregroundThreshold
    );

    const opacity = 0.7;
    const maskBlurAmount = 3;

    bodySegmentation.drawMask(
      canvas,
      image,
      binaryPartImage,
      opacity,
      maskBlurAmount
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
      <h1>People detection with TensorFlow</h1>
      <div className="prompt" id="imageResult">
        Select file with people to recognize bodies
      </div>
      <div className='input'>
          <label htmlFor='fileInput'>
          <div className='input__plus' id='filePlus' onDragOver={dropHandler}>
            +
          </div>
          </label>
        <input type="file" id="fileInput" className='input__file' onInput={() => onFileInput()}/>
        <img className="uploadedImage" id="uploadedImage" />
      </div>
      <canvas id='canvas'></canvas>
    </>
  )
}

export default App
