
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

class DrawImageMouse:
    """使用鼠标绘图"""
 
    def __init__(self, max_point=-1, line_color=(0, 0, 255), text_color=(255, 0, 0), thickness=2):
        """
        :param max_point: 最多绘图的点数，超过后将绘制无效；默认-1表示无限制
        :param line_color: 线条的颜色
        :param text_color: 文本的颜色
        :param thickness: 线条粗细
        """
        self.max_point = max_point
        self.line_color = line_color
        self.text_color = text_color
        self.focus_color = (0, 255, 0)  # 鼠标焦点的颜色
        self.thickness = thickness
        self.key = -1  # 键盘值
        self.orig = None  # 原始图像
        self.last = None  # 上一帧
        self.next = None  # 下一帧或当前帧
        self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)  # 鼠标绘制点集合
 
    def clear(self):
        self.key = -1
        self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)
        if self.orig is not None: self.last = self.orig.copy()
        if self.orig is not None: self.next = self.orig.copy()
 
    def get_polygons(self):
        """获得多边形数据"""
        return self.polygons
 
    def task(self, image, callback: Callable, winname="winname"):
        """
        鼠标监听任务
        :param image: 图像
        :param callback: 鼠标回调函数
        :param winname: 窗口名称
        :return:
        """
        self.orig = image.copy()
        self.last = image.copy()
        self.next = image.copy()
        cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(winname, callback, param={"winname": winname})
        while True:
            self.key = self.show_image(winname, self.next, delay=25)
            print("key={}".format(self.key))
            if (self.key == 13 or self.key == 32) and len(self.polygons) > 0:  # 按空格32和回车键13表示完成绘制
                break
            elif self.key == 27:  # 按ESC退出程序
                exit(0)
            elif self.key == 99:  # 按键盘c重新绘制
                self.clear()
        # cv2.destroyAllWindows()
        cv2.setMouseCallback(winname, self.event_default)
 
    def event_default(self, event, x, y, flags, param):
        pass
 
    def event_draw_rectangle(self, event, x, y, flags, param):
        """绘制矩形框"""
        if len(self.polygons) == 0: self.polygons = np.zeros(shape=(2, 2), dtype=np.int32)  # 多边形轮廓
        point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            self.next = self.last.copy()
            self.polygons[0, :] = point
            cv2.circle(self.next, point, radius=5, color=self.focus_color, thickness=self.thickness)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            print("2-EVENT_FLAG_LBUTTON")
            self.next = self.last.copy()
            cv2.circle(self.next, self.polygons[0, :], radius=4, color=self.focus_color, thickness=self.thickness)
            cv2.circle(self.next, point, radius=4, color=self.focus_color, thickness=self.thickness)
            cv2.rectangle(self.next, self.polygons[0, :], point, color=self.line_color, thickness=self.thickness)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            print("3-EVENT_LBUTTONUP")
            self.next = self.last.copy()
            self.polygons[1, :] = point
            cv2.rectangle(self.next, self.polygons[0, :], point, color=self.line_color, thickness=self.thickness)
        print("location:{},have:{}".format(point, len(self.polygons)))
 
    def event_draw_polygon(self, event, x, y, flags, param):
        """绘制多边形"""
        exceed = self.max_point > 0 and len(self.polygons) >= self.max_point
        self.next = self.last.copy()
        point = (x, y)
        text = str(len(self.polygons))
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            cv2.circle(self.next, point, radius=5, color=self.focus_color, thickness=self.thickness)
            cv2.putText(self.next, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2)
            if len(self.polygons) > 0:
                cv2.line(self.next, self.polygons[-1, :], point, color=self.line_color, thickness=self.thickness)
            if not exceed:
                self.last = self.next
                self.polygons = np.concatenate([self.polygons, np.array(point).reshape(1, 2)])
        else:
            cv2.circle(self.next, point, radius=5, color=self.focus_color, thickness=self.thickness)
            if len(self.polygons) > 0:
                cv2.line(self.next, self.polygons[-1, :], point, color=self.line_color, thickness=self.thickness)
        print("location:{},have:{}".format(point, len(self.polygons)))
 
    @staticmethod
    def polygons2box(polygons):
        """将多边形转换为矩形框"""
        xmin = min(polygons[:, 0])
        ymin = min(polygons[:, 1])
        xmax = max(polygons[:, 0])
        ymax = max(polygons[:, 1])
        return [xmin, ymin, xmax, ymax]
 
    def show_image(self, title, image, delay=5):
        """显示图像"""
        cv2.imshow(title, image)
        key = cv2.waitKey(delay=delay) if delay >= 0 else -1
        return key
 
    def draw_image_rectangle_on_mouse(self, image, winname="draw_rectangle"):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :param winname: 窗口名称
        :return: box is[xmin,ymin,xmax,ymax]
        """
        self.task(image, callback=self.event_draw_rectangle, winname=winname)
        polygons = self.get_polygons()
        box = self.polygons2box(polygons)
        return box
 
    def draw_image_polygon_on_mouse(self, image, winname="draw_polygon"):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :param winname: 窗口名称
        :return: polygons is (N,2)
        """
        self.task(image, callback=self.event_draw_polygon, winname=winname)
        polygons = self.get_polygons()
        return polygons
 

def get_linesize(length, thickness=-1, fontScale=-1.0):
    """
    自动计算绘图的大小thickness, fontScale
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=fontScale)
    """
    if fontScale <= 0: fontScale = max(2.0 * length / 850.0, 0.6)
    if thickness <= 0: thickness = int(2.0 * fontScale + 1)
    return thickness, fontScale
 
def draw_image_boxes(image, boxes, color=(0, 0, 255), thickness=-1):
    thickness, fontScale = get_linesize(max(image.shape), thickness=thickness, fontScale=-1.0)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, color, thickness=thickness)
    return image


def draw_points_on_mouse(image):
    points = []
    labels = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # left button for positive points
            points.append([x, y])
            labels.append(1)
            cv2.drawMarker(image, (x, y), (0, 255, 0), markerType=cv2.MARKER_TRIANGLE_UP,)

        if event == cv2.EVENT_RBUTTONDOWN:  # right button for negative points
            points.append([x, y])
            labels.append(0)
            cv2.drawMarker(image, (x, y), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS,)
    
    cv2.namedWindow("pick points")
    cv2.setMouseCallback("pick points", onMouse)

    while True:
        cv2.imshow("pick points", image)
        if cv2.waitKey(1)&0XFF == ord("q"):
            break
    
    cv2.destroyAllWindows()

    return points, labels
 


# Copied from https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # del mask
    # gc.collect()

def show_masks_on_image(raw_image, masks):

    if len(masks.shape) == 4:
        masks = masks.squeeze()

    nb_predictions = masks.shape[0]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, mask in enumerate(masks):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}")
        axes[i].axis("off")

    plt.show()


def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    assert not (ground_truth_map == 0).all()
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

def bbox_cropping(image, ground_truth_mask):
    x_min, y_min, x_max, y_max = get_bounding_box(ground_truth_mask)
    roi = image[y_min:y_max, x_min:x_max]
    return roi


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H = int(H * k)
    W = int(W * k)
    # H *= k
    # W *= k
    # H = int(np.round(H / 64.0)) * 64
    # W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

