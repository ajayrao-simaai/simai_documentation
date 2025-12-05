## 🛠️ Performing Graph Surgery on YOLOv8 ONNX Models
-----

## 1\. Prerequisites

Ensure you have the following files available in the specified directory structure:

| File Name | Description |
| :--- | :--- |
| `rewrite_yolov8.py` | The main script containing the logic for graph replacement/surgery. |
| `onnx_helpers.py` | A utility file containing necessary helper functions (e.g., node creation, graph manipulation). **This file must be present in the same directory** as `rewrite_yolov8.py` for the script to run correctly. |
| `your_model.onnx` | Your trained YOLOv8 model file that requires modification. |

-----

## 2\. Configuration: Editing the Script

You must modify the `rewrite_yolov8.py` script to target your specific ONNX file.

### **Step 2.1: Locate the Target Line**

Open the `rewrite_yolov8.py` file in a text editor (like `vi`, `nano`, or VS Code). Navigate to **Line 110**.

### **Step 2.2: Change the ONNX File Name**

Line 110 of the script is expected to contain the default name of the ONNX file it should load. Change this file name to match the name of your target model (`your_model.onnx`).

**Example of the change:**

| Original (Example) | Modified (Your File) |
| :--- | :--- |
| `model = onnx.load("default_yolov8n.onnx")` | `model = onnx.load("your_model_name.onnx")` |

-----

## 3\. Execution 🚀

Once the script has been updated with the correct file path, execute it using Python 3 from your terminal. Ensure your terminal's current directory is the same as the script's location (`/graph_surgery/`).

### **Step 3.1: Run the Script**

```bash
python3 rewrite_yolov8.py
```

### **Step 3.2: Expected Output**

Upon successful execution, the script will:

1.  Load the original ONNX model.
2.  Perform the required graph surgery operations (rewriting nodes).
3.  Save the modified graph to a new ONNX file.

The output file will typically be saved in the same directory, often with a suffix like `_rewritten.onnx` (e.g., `your_model_name_rewritten.onnx`), depending on the saving logic implemented in `rewrite_yolov8.py`.
