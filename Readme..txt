pip install -r requirements.txt
python detector.py "models/model.tfl" "Output/imgs_1.png" "Output/imgs_1_detections.png"


cd planesnet-detector
pip install -r requirements.txt
mkdir models
python train.py "planesnet.json" "models/model.tfl"
pip install tflearn
python train.py "planesnet.json" "models/model.tfl"
python detector.py "models/model.tfl" "scenes/scene_1.png" "scenes/scene_1_detections.png"
pip install -q tflite-model-maker --user
