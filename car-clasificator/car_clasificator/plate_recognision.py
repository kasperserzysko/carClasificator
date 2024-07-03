import easyocr
import cv2
import string

reader = easyocr.Reader(['en'], gpu=True)


def plate_del_country(plate_text):
    if plate_text.startswith("PL") and len(plate_text) != 7:
        plate_text = plate_text[3:]

    return plate_text


def process_plate(plate_img):
    resized_test_license_plate = cv2.resize(
        plate_img, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC)

    grayscale_resize_test_license_plate = cv2.cvtColor(
        resized_test_license_plate, cv2.COLOR_BGR2GRAY)

    gaussian_blur_license_plate = cv2.bilateralFilter(
        grayscale_resize_test_license_plate, 13, 15, 15)
    return gaussian_blur_license_plate


def read_license_plate(license_plate_crop):

    processed_plate = process_plate(license_plate_crop)
    ALLOWED_LIST = string.ascii_uppercase + string.digits
    detections = reader.readtext(processed_plate, allowlist=ALLOWED_LIST, width_ths=0.05)
    print("DETECTIONS: ", detections)
    plate = ""
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if score > 0.6:
            plate += text
        if len(plate_del_country(plate)) >= 5:
            return plate, score

    return None, None
