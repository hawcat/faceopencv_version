import os
import cv2
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication,QMainWindow
from PySide6.QtCore import QTimer
from ui import Ui_MainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.ui = Ui_MainWindow() #UI的实例化（）
        self.ui.setupUi(self)
        self.bind()
        self.vid = cv2.VideoCapture(0)
        self.timer = QTimer() #QT计时器
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def display_frame(self, frame):

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        resultimg = image.scaled(500,300) #绝对缩放
        self.ui.image_label.setPixmap(QPixmap.fromImage(resultimg))

    def update_frame(self):
        ret, frame = self.vid.read()
        self.display_frame(frame)

    def bind(self):
        # self.ui.___ACTION___.triggered.connect(___FUNCTION___)
        # self.ui.___BUTTON___.clicked.connect(___FUNCTION___)
        # self.ui.___COMBO_BOX___.currentIndexChanged.connect(___FUNCTION___)
        # self.ui.___SPIN_BOX___.valueChanged.connect(___FUNCTION___)
        # 自定义信号.属性名.connect(___FUNCTION___)
        self.ui.pushButton.clicked.connect(self.handle_click_pushb_caiji)
        self.ui.pushButton_2.clicked.connect(self.handle_click_pushb2_xunlian)
        self.ui.pushButton_3.clicked.connect(self.handle_click_pushb3_shibie)
        self.ui.pushButton_4.clicked.connect(self.handle_click_pushb4_jiance)

    def handle_click_pushb_caiji(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)
        face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
        # 输入id
        face_id = input('\n请输入用户ID ==>  ')
        print("\n [INFO] 正在初始化摄像头，请等待摄像头界面打开后注视摄像头保证面部可用性 ...")
        count = 0
        while (True):
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                # 保存图片
                cv2.imwrite("Data/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow('image', img)
            k = cv2.waitKey(200) & 0xff  # ESC退出and每0.2s拍一张
            if k == 27:
                break
            elif count >= 40:  # 拍40张照片
                break
        # cleanup
        print("\n [INFO] 退出采集程序，清除缓存")
        cam.release()
        cv2.destroyAllWindows()

    def handle_click_pushb2_xunlian(self):
        path = 'data'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
            return faceSamples, ids

        print("\n [INFO] 正在训练为灰度矩阵，请等待 ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        # 保存.yml训练集
        recognizer.write('Trainer/trainer.yml')
        print("\n [INFO] {0} 张脸部模型已经被训练！".format(len(np.unique(ids))))

    def handle_click_pushb3_shibie(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('Trainer/trainer.yml')
        cascadePath = "Cascades/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 0
        names = ['None', 'test1', 'test2', 'test3']
        cam = cv2.VideoCapture(0)  # 调用摄像头
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                # 判断成功概率>45时，输出id,否则输出unknow
                if (confidence < 55):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            cv2.imshow('camera', img)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        print("\n [INFO] 退出程序并清空缓存区")
        cam.release()
        cv2.destroyAllWindows()

    def handle_click_pushb4_jiance(self):
        faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # set Weight
        cap.set(4, 480)  # set Height
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
                ,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
            cv2.imshow('video', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # Esc for quit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication([]) #启动应用
    window = MainWindow() #实例化主窗口
    window.show() #展示主窗口
    app.exec()