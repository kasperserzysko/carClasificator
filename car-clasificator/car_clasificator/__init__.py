from ultralytics import YOLO
import torch
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_PATH ="C:/Users/kaspe/Desktop/sample video/sample2.mp4"

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("runs/detect/yolov8n_plate_rec3/weights/best.pt")


CAR_CLASS_ID = 2
cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))


ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        vehicle_detections = coco_model(frame)[0]
        detections_ = []
        for vehicle_detection in vehicle_detections.boxes.data.tolist(): #VEHICLE DETECTION
            veh_x1, veh_y1, veh_x2, veh_y2, conf_score, class_id = vehicle_detection
            if int(class_id) == CAR_CLASS_ID:
                detections_.append([veh_x1, veh_y1, veh_x2, veh_y2, conf_score])
                draw_border(frame, (int(veh_x1), int(veh_y1)), (int(veh_x2), int(veh_y2)), 10, line_length_x=100, line_length_y=100)
                cropped_vehicle_frame = frame[int(veh_y1):int(veh_y2), int(veh_x1):int(veh_x2), :]

                plates_detections = license_plate_detector(cropped_vehicle_frame)[0]
                for plate_detection in plates_detections.boxes.data.tolist():
                    x1, y1, x2, y2, conf_score, _ = plate_detection
                    if conf_score > 0.6:
                        print(plate_detection)
                        #draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), 10, line_length_x=100, line_length_y=100)
                        cv2.rectangle(cropped_vehicle_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                        cropped_plate = cropped_vehicle_frame[int(y1):int(y2), int(x1):int(x2), :]

                        cropped_plate_gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                        _, cropped_plate_thresh = cv2.threshold(cropped_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

cv2.destroyAllWindows()
out.release()
cap.release()
