# imports
# takes onnx -> loads -> quantize (default method) -> compiles -> saves.tar.gz
from afe.load.importers.general_importer import ImporterParams, onnx_source
from afe.apis.loaded_net import load_model
from afe.ir.tensor_type import ScalarType


# model path
onnx_model_path='Project_1 .onnx'

# input shapes dictionary: each key,value pair defines an input and its shape
input_shapes = {'images': (1, 3, 640, 640)}

# input types dictionary: each key,value pair defines an input and its type
input_types = {'images': ScalarType.float32}

# importer parameters
importer_params: ImporterParams = onnx_source(model_path=onnx_model_path,
                                            shape_dict=input_shapes,
                                            dtype_dict=input_types)

# load ONNX floating-point model into LoadedNet format
loaded_net = load_model(importer_params)

# confirm successful load
print(f"Model loaded successfully!")
print(f"LoadedNet type: {type(loaded_net)}")

# Quantization
from afe.apis.defines import default_quantization
import cv2
import numpy as np
from pathlib import Path

# Prepare calibration data from raw_data1
def generate_calibration_data(image_dir, num_samples=40):
    """Load and preprocess images for calibration"""
    image_path = Path(image_dir)
    image_files = sorted(image_path.glob('*.jpg'))[:num_samples]

    print(f"\nLoading {len(image_files)} calibration images from {image_dir}")

    for i, img_file in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_file))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        img = cv2.resize(img, (640, 640))
        # Normalize to [0, 1] and keep in HWC format (NHWC after batch dimension)
        img = img.astype(np.float32) / 255.0
        # Add batch dimension - keep as NHWC layout
        img = np.expand_dims(img, axis=0)

        yield {'images': img}

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images...")

# Generate calibration data
calib_data = generate_calibration_data('raw_data1/11zon_resized')

# Quantize the model
print("\nStarting quantization...")
quant_model = loaded_net.quantize(
    calibration_data=calib_data,
    quantization_config=default_quantization,
    model_name="yolo_defect_detection_quant"
)

print("\nModel quantized successfully!")

# Save the quantized model
model_name = "yolo_defect_detection_quant"
quant_model.save(model_name)
print(f"Quantized model saved as: {model_name}.sima")

# Compile the quantized model to .tar.gz
output_folder = "compiled_model"
print(f"\nStarting compilation to .tar.gz in '{output_folder}' folder...")
quant_model.compile(output_path=output_folder)
print(f"Model compiled successfully! Output saved to: {output_folder}/")