from ultralytics import YOLO
import torch

# Load YOLOv5/YOLOv8-style .pt model (Ultralytics handles both)
model = YOLO("Project_1 .pt")   # <-- remove space in filename!

# # Create dummy input for ONNX export
# dummy = torch.zeros(1, 3, 640, 640)

# # Export to ONNX (recommended way)
# model.export(
#     format="onnx",
#     opset=12,                      # same as your version
#     dynamic=True,                  # dynamic axes enabled
#     imgsz=640                      # input size
# )

# print("✔ Export complete: model saved as Project_1.onnx")
print(model.eval())
