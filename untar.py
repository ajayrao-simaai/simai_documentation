import tarfile

file = tarfile.open("compiled_model/yolo_defect_detection_quant_mpk.tar.gz")
for filename in file.getnames():
    print(filename)
file.close()