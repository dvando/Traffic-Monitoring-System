import cv2
import time
from traffic import TrafficCalculator

def main():
    model_path = 'models/myv4.pt'
    traffic = TrafficCalculator()

    cap = cv2.VideoCapture(0)
    # writer = cv2.VideoWriter('tes2.avi', cv2.VideoWriter_fourcc(*'mp4v'),
    #                          20, (960, 540))

    while True:
        _, frame = cap.read()
        if not _:
            print('Video finished')
            break
        tic = time.time()
        res = traffic.track(frame)
        toc = time.time()
        
        # writer.write(res)
        cv2.imshow('Result', res)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ =='__main__':
    main()