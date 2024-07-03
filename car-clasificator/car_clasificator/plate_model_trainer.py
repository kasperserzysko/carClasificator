from ultralytics import YOLO

DATA_PATH = 'D:\kasperserzysko_repository\dynaApp\carClasificator\data\plate_rec\data.yaml'
MODEL_PATH_RESUME = "D:/kasperserzysko_repository/dynaApp/carClasificator/runs/detect/yolov8n_plate_rec3/weights/last.pt"
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
