import cv2


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


def display_plate(frame, cropped_plate, license_plate_text, veh_y1, veh_x2, veh_x1):
    H, W, _ = cropped_plate.shape
    try:
        frame[int(veh_y1) - H - 100:int(veh_y1) - 100, int((veh_x2 + veh_x1 - W) / 2):int((veh_x2 + veh_x1 + W) / 2), :] = cropped_plate

        frame[int(veh_y1) - H - 400:int(veh_y1) - H - 100, int((veh_x2 + veh_x1 - W) / 2):int((veh_x2 + veh_x1 + W) / 2), :] = (255, 255, 255)

        (text_width, text_height), _ = cv2.getTextSize(
            license_plate_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            12)

        cv2.putText(frame,
                    license_plate_text,
                    (int((veh_x2 + veh_x1 - text_width) / 2), int(veh_y1 - H - 250 + (text_height / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    12)
    except:
        pass


def display_plate_image(frame, cropped_plate, license_plate_text, y1, x2, x1):
    H, W, _ = cropped_plate.shape
    if license_plate_text == None:
        license_plate_text = "DUPA"

    (text_width, text_height), _ = cv2.getTextSize(
        license_plate_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        6)
    cv2.putText(frame,
                license_plate_text,
                (int((x2 + x1 - text_width) / 2), int(y1 - H - 50 + (text_height / 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                6)