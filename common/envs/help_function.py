import cv2

def cvt_gray_resize(frame, height = 84, width = 84):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), 
                    interpolation=cv2.INTER_AREA)
    return frame