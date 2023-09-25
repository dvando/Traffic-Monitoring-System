import cv2
import numpy as np
import time
from ultralytics import YOLO

class TrafficCalculator:
    """A Class to wrap YOLO model and calculate the traffic stats"""
    
    def __init__(self, model_path='models/myv4.pt'):
        self.cur_num = 0
        self.cur_stat = 'Lancar'
        self.cur_speed_avg = 0
        self.cur_t = 0
        self.temp_t = time.time()
        self.model = YOLO(model_path)
        self.stats = {}
        self.temp_stats = {}
        self.constant = 0.2
        self.line_coords = None
        self.num_chart = np.zeros((60,))
        self.speed_chart = np.zeros((60,))
    
    def track(self, frame):
        """Main method to track the frame

        Args:
            frame (cv Mat): input image

        Returns:
            cv Mat: Plotted image
        """
        # frame = cv2.resize(frame, (960,540))
        if not self.line_coords:
            h, w = frame.shape[:2]
            self.line_coords = ((0, h*2//5), (w, h*2//5))
            
        result = self.model.track(frame, classes=2, persist=True,
                          verbose=False, show_conf=False)
        
        res = result[0].plot()
        boxes = result[0].boxes.xywh
        track_id = result[0].boxes.id
        
        if boxes is not None:
            centroids = [self.centroid(box) for box in boxes.cpu().numpy()]
        
        if track_id is not None:
            for i, track in enumerate(track_id.cpu().numpy()):
                self.stats[track] = (centroids[i], time.time())
        
        self.calc_stats()        
        res = self.plot(res)

        return res
    
    def centroid(self, xywh):
        """Calculate centroid based on xywh box

        Args:
            xywh (ndarray): box in xywh format

        Returns:
            ndarray: centroid coordinate (x, y)
        """
        
        return np.array([xywh[0]+xywh[2]//2, xywh[1]+xywh[3]//2])
    
    def euclidd(self, centroid1, centroid2):
        """Calculate Euclidean Distance of 2 centroids

        Args:
            centroid1 (ndarray): centroid 1 (x, y)
            centroid2 (ndarray): centroid 2 (x, y)

        Returns:
            ndarray: distance
        """
        
        return np.sqrt((centroid1[0]-centroid2[0])**2 + (centroid1[1]-centroid2[1])**2)
    
    def _calc_speed(self, distance, t_frame=None):
        """Internal method to calculate speed"""
        
        return distance/t_frame * self.constant
    
    def _update_stat(self):
        """Method to update current traffic status"""
        if self.cur_num > 30 and self.cur_speed_avg < 20:
            self.cur_stat = "Padat"
        elif self.cur_num > 20 and self.cur_speed_avg > 20:
            self.cur_stat = "Merayap"
        else:
            self.cur_stat = "Lancar"
        
    def calc_stats(self):
        """Method to calculate and update all traffic statistics"""
        if len(self.stats) == 0:
            return None
        else:
            self.cur_num = len(self.stats)
            
        speeds = np.array([self._calc_speed(self.euclidd(self.stats[id][0], self.temp_stats[id][0]), self.stats[id][1] - self.temp_stats[id][1]) for id in self.stats.keys() if id in self.temp_stats.keys()])
        if len(speeds) != 0:
            self.cur_speed_avg = speeds.sum() / len(speeds)
        
        self.cur_t = time.time()
        t_elapsed = self.cur_t - self.temp_t
        
        if t_elapsed > 60:
            self.num_chart[:-1] = self.num_chart[1:]
            self.num_chart[-1] = self.cur_num
            
            self.speed_chart[:-1] = self.speed_chart[1:]
            self.speed_chart[-1] = self.cur_speed_avg
        
            self.temp_t = time.time()
            
            # print(f'num: {self.num_chart}')
            # print(f'spd: {self.speed_chart}')
        self._update_stat()
        self.temp_stats = self.stats.copy()
        self.stats.clear()
            
    def plot(self, img):
        """Method to plot status into the image"""
        res = cv2.putText(img, f'Status: {self.cur_stat}', (25, 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                          2, cv2.LINE_AA)
        
        res = cv2.putText(res, f'Total Car: {self.cur_num}', (25, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                          2, cv2.LINE_AA)

        res = cv2.putText(res, f'Average Speed: {int(self.cur_speed_avg)}', (25, 75), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                          2, cv2.LINE_AA)
        
        return res
    
    def update_streamlit(self, placeholder, data):
        placeholder.line_chart(data)
