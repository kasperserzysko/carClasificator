import cv2
from ultralytics import YOLO    #toDO to  wyrzucic (uzyc pytorch)
from util import draw_border, display_plate, display_plate_image
from plate_recognision import read_license_plate

coco_model = YOLO('yolov8n.pt')     #PATRZ
license_plate_detector = YOLO("runs/detect/yolov8n_plate_rec3/weights/best.pt")     #PATRZ


CAR_CLASS_ID = 2


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

                    plates_detections = license_plate_detector(cropped_vehicle_frame)[0]
                    for plate_detection in plates_detections.boxes.data.tolist():
                        x1, y1, x2, y2, conf_score, _ = plate_detection
                        if conf_score > 0.6:
                            cv2.rectangle(cropped_vehicle_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                            cropped_plate = cropped_vehicle_frame[int(y1):int(y2), int(x1):int(x2), :]

                            license_plate_text, license_plate_text_score = read_license_plate(cropped_plate)
                            if license_plate_text is not None:
                                display_plate(frame, cropped_plate, license_plate_text, veh_y1, veh_x2, veh_x1)
                    # TODO dorzuc tutaj marke auta

            out.write(frame)
            cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()


def image_process(in_path, out_path):
    image = cv2.imread(in_path)

    plates_detections = license_plate_detector(image)[0]
    for plate_detection in plates_detections.boxes.data.tolist():
        x1, y1, x2, y2, conf_score, _ = plate_detection
        if conf_score > 0.6:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            cropped_plate = image[int(y1):int(y2), int(x1):int(x2), :]

            license_plate_text, license_plate_text_score = read_license_plate(cropped_plate)
            if license_plate_text is not None:
                 print("PLATE: ", license_plate_text)
            display_plate_image(image, cropped_plate, license_plate_text, y1, x2, x1)
            # TODO dorzuc tutaj marke auta
    cv2.imwrite(out_path + ".jpg", image)
