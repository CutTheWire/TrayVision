import cv2
from pygrabber.dshow_graph import FilterGraph

class Camera:
    def __init__(self):
        self.available_cameras = {}
        self.devices = FilterGraph().get_input_devices()
        self.camera_name = 'HD 4MP WEBCAM'
        self.cap = None
        self.name = ""
        
    def get_available_cameras(self):
        self.available_cameras = {}
        for device_index, device_name in enumerate(self.devices):
            self.available_cameras[device_index] = device_name
    
    def cameras_list(self):
        self.get_available_cameras()
        return self.available_cameras
    
    def open_camera(self):
        available_cameras = self.cameras_list()
        for i in available_cameras:
            if available_cameras[i] == self.camera_name:
                self.cap = cv2.VideoCapture(i)
                self.name = available_cameras[i]
            elif self.cap == None:
                self.cap = cv2.VideoCapture(0)
                self.name = available_cameras[0]

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', 'E', 'V', 'C'))
        return self.cap
    
    def set_cap_size(self, width: int, height: int):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()

    def is_camera_open(self):
        return self.cap is not None

    def __del__(self):
        self.release_camera()
