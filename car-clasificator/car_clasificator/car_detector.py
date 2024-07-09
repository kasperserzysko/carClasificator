import cv2
from ultralytics import YOLO
from util import draw_border, display_plate, display_plate_image
from plate_recognision import read_license_plate

import torch
from torchvision import transforms, models

from PIL import Image
import torch.nn as nn

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("your_model_path.pt")     #wstaw sciezke do swojego modelu
resnet_model_path = 'utilities/car_brand_classifier_resnet.pth'

CAR_CLASS_ID = 2
#transoframcja danych do wykorzystania przy szukaniu marek samochodow
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

car_brands = ['alfa romeo', 'Audi', 'Bentley', 'Benz','Bmw', 'Cadillac', 'Dodge', 'Ferrari', 'Ford','Ford mustang', 'hyundai', 'Kia','Lamborghini', 'Lexus', 'Maserati', 'Porsche', 'Rolls royce', 'Tesla', 'Toyota' ]

model_path = '../../output/car_brand_classifier_resnet.pth'

def load_model(model_path, num_classes):
    # zaladowanie modelu ResNet50 za pomoca sciezki do pliku
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model
model = load_model(model_path,19)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def predict_car_brand(image, model, transform, device):
    #sprawdzamy czy zostala przeslana sciezka pliku do zdjecia do rozpoznania czy klatka wycieta z filmu
    if isinstance(image, str):
        image = Image.open(image)
    else:
        image = Image.fromarray(image)

    # zamina PIL image na tensor
    image = transform(image).unsqueeze(0).to(device)

    #predykcja marki samochodu
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

    return predicted_label

def video_process(in_path, out_path):

    cap = cv2.VideoCapture(in_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path + ".mp4", fourcc, fps, (width, height))

    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            vehicle_detections = coco_model(frame)[0]
            for vehicle_detection in vehicle_detections.boxes.data.tolist():  # VEHICLE DETECTION
                veh_x1, veh_y1, veh_x2, veh_y2, conf_score, class_id = vehicle_detection
                if int(class_id) == CAR_CLASS_ID:
                    draw_border(frame, (int(veh_x1), int(veh_y1)), (int(veh_x2), int(veh_y2)), 10, line_length_x=100,
                                line_length_y=100)
                    cropped_vehicle_frame = frame[int(veh_y1):int(veh_y2), int(veh_x1):int(veh_x2), :]
                    car_brand_idx = predict_car_brand(cropped_vehicle_frame, model, transform, device)
                    car_brand = car_brands[car_brand_idx]
                    cv2.putText(frame, car_brand, (int(veh_x1) - 20, int(veh_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 0), 2)
                    plates_detections = license_plate_detector(cropped_vehicle_frame)[0]
                    for plate_detection in plates_detections.boxes.data.tolist():
                        x1, y1, x2, y2, conf_score, _ = plate_detection
                        if conf_score > 0.6:
                            cv2.rectangle(cropped_vehicle_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                            cropped_plate = cropped_vehicle_frame[int(y1):int(y2), int(x1):int(x2), :]



                            license_plate_text, license_plate_text_score = read_license_plate(cropped_plate)
                            if license_plate_text is not None:
                                display_plate(frame, cropped_plate, license_plate_text, veh_y1, veh_x2, veh_x1)


            out.write(frame)
            cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()


def image_process(in_path, out_path):
    image = cv2.imread(in_path)

    plates_detections = license_plate_detector(image)[0]
    car_brand_idx = predict_car_brand(in_path, model, transform, device)
    car_brand = car_brands[car_brand_idx]
    x1, y1, _ = image.shape
    cv2.putText(image, car_brand, (int(x1)-200, int(y1) - 405), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),
                2)
    for plate_detection in plates_detections.boxes.data.tolist():
        x1, y1, x2, y2, conf_score, _ = plate_detection
        if conf_score > 0.6:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            cropped_plate = image[int(y1):int(y2), int(x1):int(x2), :]

            license_plate_text, license_plate_text_score = read_license_plate(cropped_plate)
            if license_plate_text is not None:
                 print("PLATE: ", license_plate_text)
            display_plate_image(image, cropped_plate, license_plate_text, y1, x2, x1)
          
    cv2.imwrite(out_path + ".jpg", image)
