

class ImgData:
    def __init__(self, filename, width, height):
        self.filename = filename
        self.width = width
        self.height = height
        self.imageset = ''
        self.bboxes = []


class DetectData:
    def __init__(self, class_name, x1, x2, y1, y2):
        self.class_name = class_name
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
