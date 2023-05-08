## Automatic number plate recogition (Python)

This repo contains implemetation of ANPR built using Python

### The process of ANPR in this project requires 3 majot block

- Car Detection
- Licence plate detection
- Character recognition

### 1. Car Detection
This is acheived using pretrained yolo model.

### 2. Licence plate detection
This is acheived by using pretrained model from the mentioned Github repo as collecting and training own model requires time.

Model path : [Number plate revognition](https://github.com/wasdac9/automatic-number-plate-recognition/blob/main/best.pt_)

### 3. Character recognition
To achieve this step, Paddle OCR is used. Paddle ocr gives state of the performace when compared to other open source model.

## Steps to replicate the project

- git clone -b v1.0 https://github.com/Monalsingh/ANPR.git
- docker pull ultralytics/ultralytics
- docker run --name ultralytics_docker --gpus=all -it -v /home/project:/home/project --shm-size=24g ultralytics/ultralytics

- nvidia-smi
- apt update
- apt install -y zip htop screen libgl1-mesa-glx
- conda create -n env pytho=3.8
- pip install opencv-python==4.5.5.64
- pip install "numpy<1.24"
- pip install ultralytics
- pip install "paddleocr>=2.0.1"
- python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
- export LD_LIBRARY_PATH=/opt/conda/envs/assert_task/lib

### Code to run the ANPR
```bash
python main.py
```
Open main.py and add the path to image/video folder. It will run inference on all files present inside input directory and create a single output video with all the inference ouput combined.

Update the path to model present inside Class initializer.

OCR inference is running on CPU not GPU, cuda needs to be configure to run OCR inference on GPU, by default it is set to CPU.

