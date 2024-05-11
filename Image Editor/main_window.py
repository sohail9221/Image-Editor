import sys
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow,QStatusBar , QWidget, QFileDialog, QPushButton, QGridLayout, QComboBox, QLineEdit
from PySide6.QtCore import Qt
import numpy as np
from PIL import Image, ImageOps
import cv2
import numpy as np
from sklearn.cluster import KMeans


class ImageProcessor(QMainWindow):
    def __init__(self, app):
        super(ImageProcessor, self).__init__()

        self.app = app
        self.showMaximized()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Image Processor")
        self.Menu()
        self.StatusBar()
        self.showMaximized()
    
        self.original_video = None
        self.processed_video = None

        self.total_frames = 0
        self.current_frame = 0

        self.original_image_path = "image\image.jpg"
        self.processed_image_path =  "image\image.jpg"

        self.original_image = np.array(cv2.imread(self.original_image_path))
       # self.original_image = cv2.imread(self.original_image_path)
        self.processed_image = np.array(cv2.imread(self.processed_image_path))
        
        self.MainBar()

        if self.current_frame == 0:
            self.adjustSize()

    def Menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&Upload")
        upload_image_action = file_menu.addAction("Upload New Image")
        upload_video_action = file_menu.addAction("Upload New Video")

        quit = self.menuBar()
        quit_action = quit.addAction("Quit")

        upload_image_action.triggered.connect(self.upload_image)
        upload_video_action.triggered.connect(self.upload_video)
        quit_action.triggered.connect(self.quit)

    def MainBar(self): 
        self.grid_layout = QGridLayout(self.central_widget)
        process_image = QPushButton("Process Image")
        process_image.setFixedSize(150, 50)

        self.previous_button = QPushButton("Previous Frame")
        self.previous_button.setFixedSize(150, 50)
        self.previous_button.setVisible(False)

        self.next_button = QPushButton("Next Frame")
        self.next_button.setFixedSize(150, 50)
        self.next_button.setVisible(False)


        self.download_image = QPushButton("Download Image")
        self.download_image.setFixedSize(150, 50)
        self.download_image.setVisible(False)

        self.download_frame = QPushButton("Download Frame")
        self.download_frame.setFixedSize(150, 50)
        self.download_frame.setVisible(False)


        self.combo_box = QComboBox(self)
        self.combo_box.addItem("Image Negative")
        self.combo_box.addItem("Log Transformation")
        self.combo_box.addItem("Power Law Transformation")
        self.combo_box.addItem("Histogram Equalization")
        self.combo_box.addItem("Averaging Filter")
        self.combo_box.addItem("Laplacian")
        self.combo_box.addItem("Adaptive Thresholding")
        self.combo_box.addItem("Segmentation")
        self.combo_box.addItem("Clustering")
        self.combo_box.addItem("LoG")
        self.combo_box.addItem("Erosion")
        self.combo_box.addItem("Dilation")
        self.combo_box.addItem("Opening")
        self.combo_box.addItem("Closing")
        self.combo_box.addItem("cv2 erosion")
        self.combo_box.addItem("cv2 dilation")
        self.combo_box.addItem("cv2 opening")
        self.combo_box.addItem("cv2 closing")

        self.combo_box.setFixedSize(150, 50)

        frame = self.original_image
        if len(frame.shape) == 3:
            height, width, channel= frame.shape
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width

        self.image = QLabel()
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image.setPixmap(pixmap)

        frame = self.processed_image
        if len(frame.shape) == 3:
            height, width, channel= frame.shape
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width

        self.image2 = QLabel()
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image2.setPixmap(pixmap)

        technique = QLabel("Select a Processing Technique")

        self.log_input = QLineEdit()
        self.log_input.setPlaceholderText("r = ")
        self.log_input.setVisible(False)
        self.log_input.setFixedSize(150, 20)

        self.power_input = QLineEdit()
        self.power_input.setPlaceholderText("gamma = ")
        self.power_input.setVisible(False)
        self.power_input.setFixedSize(150, 20)

        self.average = QLineEdit()
        self.average.setPlaceholderText("kernel size = ")
        self.average.setVisible(False)
        self.average.setFixedSize(150, 20)

        self.combo_box.currentIndexChanged.connect(lambda: self.combo_box_index_changed(self.combo_box))
        process_image.clicked.connect(lambda: self.process())
        self.previous_button.clicked.connect(lambda: self.previous_frame())
        self.next_button.clicked.connect(lambda: self.next_frame())


        self.grid_layout.addWidget(self.image, 0, 6, 4, 1)
        self.grid_layout.addWidget(self.image2, 0, 8, 4, 1)
        self.grid_layout.addWidget(technique, 0, 0)
        self.grid_layout.addWidget(self.combo_box, 1, 0)
        self.grid_layout.addWidget(self.log_input, 2, 0)
        self.grid_layout.addWidget(self.power_input, 2, 0)
        self.grid_layout.addWidget(self.average, 2, 0)

        self.grid_layout.addWidget(process_image, 4, 0)
        self.grid_layout.addWidget(self.previous_button, 4, 6)
        self.grid_layout.addWidget(self.next_button, 4, 8)
        self.grid_layout.addWidget(self.download_image, 4, 2)
        self.grid_layout.addWidget(self.download_frame, 4, 2)

        self.download_image.clicked.connect(lambda: self.download_images())
        self.download_frame.clicked.connect(lambda: self.download_video())

    def download_images(self):
        self.save_image(self.processed_image)

    def download_video(self):
        self.save_image(self.processed_video[self.current_frame])

    def combo_box_index_changed(self, combo_box):
        if combo_box.currentIndex() == 1:
            self.log_input.setVisible(True)
        elif combo_box.currentIndex() == 2:
            self.power_input.setVisible(True)
        elif combo_box.currentIndex() == 4  :
            self.average.setVisible(True)
        elif combo_box.currentIndex() == 10:
            self.power_input.setVisible(True)
        elif combo_box.currentIndex() == 11  :
            self.average.setVisible(True)
        elif combo_box.currentIndex() == 12:
            self.power_input.setVisible(True)
        elif combo_box.currentIndex() == 13  :
            self.average.setVisible(True)   
       
        elif combo_box.currentIndex() == 14  :
            self.average.setVisible(True)  
        elif combo_box.currentIndex() == 15:
            self.power_input.setVisible(True)
           
        else:
            self.power_input.setVisible(False)
            self.log_input.setVisible(False)
            self.average.setVisible(False)

    def display_frame(self):
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
    
        frame = self.processed_video[self.current_frame]
        height, width, channel= frame.shape
        bytes_per_line = 3 * width

        self.image3 = QLabel()
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image3.setPixmap(pixmap)

        frame = self.original_video[self.current_frame]
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        self.image4 = QLabel()
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image4.setPixmap(pixmap)

        self.grid_layout.addWidget(self.image4, 0, 6, 4, 1)
        self.grid_layout.addWidget(self.image3, 0, 8, 4, 1)
        self.setMaximumWidth(1000)
        
    def process(self):
        if self.original_video == None:
            self.process_image()
        else:
            self.process_video()

    def process_image(self):
        index = self.combo_box.currentIndex() 
        if index == 0:
            
            self.processed_image = self.negative_image(self.original_image)
            print("Negative Function for Image")
        elif index == 1:
            
            r_value = float(self.log_input.text())
            self.processed_image = self.log_transform(self.original_image,r_value)
            print("Log Function for Image")
        elif index == 2:
            
            gamma_value = float(self.power_input.text())
            self.processed_image = self.power_transform(self.original_image,gamma_value)
            print("Power Function for Image")
        elif index == 3:
            
            self.processed_image = self.histogram_transform(self.original_image)
            print("Histogram Function for Image")
        elif index == 4:
            kernel_size = int(self.average.text())
            self.processed_image = self.averaging_filter(self.original_image,kernel_size)
            print("Averaging Function for Image")
        elif index==5:
            
            self.processed_image = self.laplacian(self.original_image)
            print("Laplacian Function for Image")
        elif index==6:
                self.processed_image = self.adaptive_thresholding(self.original_image)
                print("Adaptive threshholding for Image")
                
        elif index == 7:
            #do segmentation
            self.processed_image = self.segmentation(self.original_image)
            print("Segmentation Applied on the Image")
        elif index==8:
            #do clustering
            self.processed_image = self.ImageClustering(self.original_image)
            print("Clustering Applied on the Image on Optimal Number of Clusters")
            a=6

        elif index == 9:
            #do Log   
            
            self.processed_image = self.apply_log_filter(self.original_image)
            print("LOG Applied on the image")
             
        elif index==10:
            kernel_size = int(self.average.text())
            self.processed_image = self.erosion(self.original_image, kernel_size)
            print("erosion for Image")
        elif index==11:
                kernel_size = int(self.average.text())
                self.processed_image = self.dilation(self.original_image, kernel_size)
                print("dilation for Image")
                
        elif index == 12:
           
            kernel_size = int(self.average.text())
            self.processed_image = self.opening(self.original_image, kernel_size)
            print("openingon the Image")
        elif index == 13 :
           
            kernel_size = int(self.average.text())
            self.processed_image = self.closing(self.original_image, kernel_size)
            print("Closing on image")

        elif index == 14:
            kernel_size = int(self.average.text())
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        # Create erosion kernel with specified size
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = cv2.erode(self.original_image, kernel)
            print("cv2 erosion Applied on the Image")
        elif index == 15:
            kernel_size = int(self.average.text())
            if kernel_size % 2 == 0:
                    kernel_size += 1
    
    # Create dilation kernel with specified size
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
            center = kernel_size // 2
            kernel[center, center] = 1
            self.processed_image = cv2.dilate(self.original_image, kernel)
            print("cv2 dilation Applied on the Image")
        elif index == 16:
            
            kernel_size = int(self.average.text())
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        # Create erosion kernel with specified size
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
            print("cv2 opening Applied on the Image")
        elif index == 17:
            kernel_size = int(self.average.text())
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        # Create erosion kernel with specified size
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
            print("cv2 closing Applied on the Image")    



                             
              
        frame = self.processed_image
        if len(frame.shape) == 3:
            height, width, channel= frame.shape
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width

        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image2.setPixmap(pixmap)
    
    def process_video(self):
        index = self.combo_box.currentIndex() 
        image = np.copy(self.processed_video[self.current_frame])
        
        if index == 0:
            self.processed_video[self.current_frame] = self.negative_image(image)
        elif index == 1:
            r_value = float(self.log_input.text())
            self.processed_video[self.current_frame] = self.log_transform(image,r_value)
        elif index == 2:
            gamma_value = float(self.power_input.text())
            self.processed_video[self.current_frame] = self.power_transform(image,gamma_value)
        elif index == 3:
            self.processed_video[self.current_frame] = self.histogram_transform(image)
        elif index == 4:
            kernel_size = int(self.average.text())
            self.processed_video[self.current_frame] = self.averaging_filter(image,kernel_size)
        elif index==5:
            self.processed_video[self.current_frame] = self.laplacian(image)

           
        elif index == 6:
             self.processed_video[self.current_frame] = self.adaptive_thresholding(image)
             print("Adaptive threshholding for Image")
                
        elif index == 7:
            #do segmentation
            
            self.processed_video[self.current_frame] = self.segmentation(image)
        elif index==8:
            #do clustering
            self.processed_video[self.current_frame] = self.ImageClustering(image)
            print("Clustering Applied on the Image on Optimal Number of Clusters")
            a=6
        elif index == 9:
            #do Log   
            
            self.processed_video[self.current_frame] = self.apply_log_filter(image)
            print("LOG Applied on the image")
        elif index==10:
            kernel_size = int(self.average.text())
            self.processed_video[self.current_frame]= self.erosion(image, kernel_size)
            print("erosion for video")
        elif index==11:
                kernel_size = int(self.average.text())
                self.processed_video[self.current_frame] = self.dilation(image, kernel_size)
                print("dilation for video")
                
        elif index == 12:
           
            kernel_size = int(self.average.text())
            self.processed_video[self.current_frame] = self.opening(image, kernel_size)
            print("opening on video")
        else :
           
            kernel_size = int(self.average.text())
            self.processed_video[self.current_frame] = self.closing(image, kernel_size)
            print("closing on video")
                                 

        self.display_frame()
    
    def StatusBar(self):
        self.setStatusBar(QStatusBar(self))

    def upload_image(self):
        self.download_frame.setVisible(False)
        self.download_image.setVisible(True)
        
        frame = np.array(Image.open(self.original_image_path))

        if len(frame.shape) == 3:
            height, width, channel= frame.shape
            
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image.setPixmap(pixmap)
       
        frame = np.array(Image.open( self.original_image_path))
        if len(frame.shape) == 3:
            height, width, channel= frame.shape
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image2.setPixmap(pixmap)

        try:

            self.original_video = None
            self.processed_video = None
           
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image files (*.png; *.jpg; *.jpeg; *.gif)')
            self.original_image = np.array(Image.open(file_path))
            self.statusBar().showMessage(f"You uploaded an Image ")
            frame = self.original_image
            height, width, channel= frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image.setPixmap(pixmap)

            self.previous_button.setVisible(False)
            self.next_button.setVisible(False)

        except:
          
            pass

        self.adjustSize()
        #self.adjust_frame_size()

    def upload_video(self):
        self.download_frame.setVisible(True)
        self.download_image.setVisible(False)
        frame = np.array(Image.open("image\image.jpg"))
        if len(frame.shape) == 3:
            height, width, channel= frame.shape
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image.setPixmap(pixmap)

        frame = np.array(Image.open("image\image.jpg"))
        if len(frame.shape) == 3:
            height, width, channel= frame.shape
        else:
            height, width= frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image2.setPixmap(pixmap)
        #self.adjust_frame_size()

        try:
            self.image.clear()
            self.image2.clear()
            self.original_image = None
            self.processed_image = None
            self.statusBar().showMessage("You uploaded a Video")
            video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video files (*.mp4; *.avi; *.mkv)')
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = []
            frames2 = []
            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                frames.append(frame)
                frames2.append(frame)

            cap.release()

            self.original_video = frames
            self.processed_video = frames2
            self.total_frames = total_frames
            self.current_frame = 0
            self.previous_button.setVisible(True)
            self.next_button.setVisible(True)
            self.display_frame()
        except:
            pass

        #self.adjustSize()

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame >= len(self.original_video):
            self.current_frame = 0

        self.process_video()
    
    def previous_frame(self):
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = self.total_frames - 1

        self.process_video()

    def quit(self):
        self.app.quit()

    # def adaptive_thresholding(self,image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv2.THRESH_BINARY, block_size=11, c=2):
 
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     return cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, c)    


    def adaptive_thresholding(self,image, block_size=11, c=2):
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # Initialize the output image
        thresholded = np.zeros_like(gray)

        # Iterate over each pixel in the image
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                # Define the region of interest
                roi = gray[max(0, y - block_size // 2):min(gray.shape[0], y + block_size // 2 + 1),
                        max(0, x - block_size // 2):min(gray.shape[1], x + block_size // 2 + 1)]

                # Calculate the local mean and standard deviation
                local_mean = np.mean(roi)
                local_std = np.std(roi)

                # Compute the threshold value using the local mean and standard deviation
                threshold = local_mean - c * local_std

                # Apply thresholding
                if gray[y, x] > threshold:
                    thresholded[y, x] = 255
                else:
                    thresholded[y, x] = 0

        return thresholded

# Example usage:
# thresholded_image = adaptive_thresholding(input_image, block_size=11, c=2)


    def negative_image(self,image):
        self.statusBar().showMessage(f"In the Negative Function")
        max_intensity = np.max(image)
        image = max_intensity - image
        return image

    def log_transform(self, image, r_value):
        self.statusBar().showMessage(f"In the Log Function")

        image = image.astype(np.float32) / 255.0
        log_transformed = np.log1p(r_value * image) / np.log1p(r_value)
        log_transformed *= 255.0
        return np.clip(log_transformed, 0, 255).astype(np.uint8)

    def power_transform(self, image, gamma_value):
        self.statusBar().showMessage(f"In the Power Function")
        image = image.astype(np.float32) / 255.0
        gamma_corrected = np.power(image, gamma_value)
        gamma_corrected *= 255.0
        return np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    def histogram_transform(self, image):
        self.statusBar().showMessage("In the Histogram Function")

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized_hist = cv2.equalizeHist(image_gray)
            equalized_result = cv2.cvtColor(equalized_hist, cv2.COLOR_GRAY2BGR)
        else:
            equalized_result = cv2.equalizeHist(image)

        return equalized_result
    
    def averaging_filter(self,image,kernel_size = 6):
        self.statusBar().showMessage("In the Averaging Function")
        image = image.astype(np.float32) / 255.0
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        filtered_image = cv2.filter2D(image, -1, kernel)
        filtered_image *= 255.0
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    def laplacian(self, image):
        self.statusBar().showMessage("In the Laplacian Function")

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        laplacian_result = cv2.Laplacian(image_gray, cv2.CV_64F)
        laplacian_result = np.clip(laplacian_result, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            laplacian_result = cv2.cvtColor(laplacian_result, cv2.COLOR_GRAY2BGR)

        return laplacian_result
    

    def segmentation(self, image):
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a black image with the same dimensions as the input image
        segmented_image = np.zeros_like(np.array(image))
        
        # Draw contours on the segmented image
        cv2.drawContours(segmented_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        return segmented_image


    def ImageClustering(self, image, num_clusters=3):
        # Convert the image to a numpy array
        image_array = np.array(image)
    
        image_flat = image_array.reshape((-1, 3))
        
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        labels = kmeans.fit_predict(image_flat)
        clustered_image_flat = np.reshape(labels, (image_array.shape[0], image_array.shape[1]))
    
        segmented_image = np.zeros_like(image_array)
        
  
        colors = kmeans.cluster_centers_
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                segmented_image[i, j] = colors[clustered_image_flat[i, j]]
        
        return segmented_image

    def apply_log_filter(self, image, kernel_size=(5, 5), sigma=1.0):
        try:
            
            image_np = np.array(image)
            
            # Ensure image is not None
            if image_np is None:
                raise ValueError("Input image is None.")
            
            if len(image_np.shape) == 3:  
                gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_np  
            gray_image = np.uint8(gray_image)
            
            # Apply Laplacian of Gaussian (LOG) filter
            filtered_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)
            filtered_image = cv2.Laplacian(filtered_image, cv2.CV_64F)
            
            # Normalize the filtered image to values between 0 and 255
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            _, binary_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            segmented_image = np.zeros_like(image_np)
            
            # Draw contours on the segmented image
            cv2.drawContours(segmented_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            
            return segmented_image
        except Exception as e:
            print("Error in apply_log_filter:", e)
            return None
   
    


    def erosion(self,image, kernel_size=3):
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        rows, cols, channels = image.shape
        
        output = np.zeros_like(image)
        
        k_center = kernel_size // 2
        
        for i in range(k_center, rows - k_center):
            for j in range(k_center, cols - k_center):
                for c in range(channels):
                    neighborhood = image[i - k_center:i + k_center + 1, j - k_center:j + k_center + 1, c]
                    output[i, j, c] = np.min(neighborhood)
        
        return output



    def dilation(slef,image, kernel_size=3):
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        rows, cols, channels = image.shape
        
        output = np.zeros_like(image)
        
        k_center = kernel_size // 2
        
        for i in range(k_center, rows - k_center):
            for j in range(k_center, cols - k_center):
                for c in range(channels):
                    neighborhood = image[i - k_center:i + k_center + 1, j - k_center:j + k_center + 1, c]
                    output[i, j, c] = np.max(neighborhood)
        
        return output





    def opening(self,image, kernel):
        # Perform erosion followed by dilation
        return self.dilation(self.erosion(image, kernel), kernel)

    def closing(self,image, kernel):
        # Perform dilation followed by erosion
        return self.erosion(self.dilation(image, kernel), kernel)

    def adjustSize(self):
        max_width = 1200
        if self.width() > max_width:
            self.setFixedWidth(max_width)

    def save_image(self, image):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Image files (*.png; *.jpg; *.jpeg)')
        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor(app)
    window.show()
    sys.exit(app.exec())
