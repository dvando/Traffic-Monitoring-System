import streamlit as st
import cv2
import av
import sys
import argparse
from traffic import TrafficCalculator
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-webcam', type=bool, default=False)
    parser.add_argument('--vid-path', type=str, default='assets/tes1.mp4')
    
    return parser.parse_args(sys.argv[1:])

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.traffic = TrafficCalculator()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        
        res = self.traffic.track(
            img, 
            classes=0, 
            persist=True, 
            verbose=False
            )
        
        plotted = res[0].plot()
        
        return plotted

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    use_webcam = True
    traffic = TrafficCalculator()
    
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        res = frame.to_ndarray(format="bgr24")
        res = traffic.track(res)
        
        return av.VideoFrame.from_ndarray(res, format="bgr24")
    
    st.set_page_config(
        page_title="Traffic Monitoring System",
        page_icon="✅",
        layout="wide",
    )

    st.title("Real-Time / Live Traffic Monitoring Dashboard Based on YOLOv4")


    st.header('Live Stream')
    if use_webcam:
        webrtc_streamer(
                key="my_traffic",
                video_frame_callback=video_frame_callback,
                rtc_configuration={  # Add this config
				"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			    },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
    else:
        st.video(args.vid_path)

    st.header("Stats")
    col1, col2 = st.columns(2)
    
    with col1:    
        st.subheader('Average Number of Car per Hour')
        lc = st.empty()
        traffic.update_streamlit(lc, traffic.num_chart)
        # lc.line_chart(traffic.num_chart)

    with col2:            
        st.subheader('Average Car Speed per Hour')
        sc = st.empty()
        traffic.update_streamlit(sc, traffic.num_chart)
        # sc.line_chart(traffic.speed_chart)
