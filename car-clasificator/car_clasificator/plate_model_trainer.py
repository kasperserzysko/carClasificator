from ultralytics import YOLO

DATA_PATH = 'data_for_plate_recognition.yaml' #wstawic sciezke do danych treningowych dla yolo v8
MODEL_PATH_RESUME = "your_model_path.pt"    #wstaw sciezke do swojego modelu
MODEL_PATH = "yolov8n.pt"

# Load a model
model = YOLO(MODEL_PATH)

   # Use the model
results = model.train(
   data=DATA_PATH,
   imgsz=1280,
   epochs=30,
   batch=8,
   name='yolov8n_plate_rec')

   #results = model.train(resume=True)
