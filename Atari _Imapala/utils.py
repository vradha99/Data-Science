import shutil
import datetime
from cv2 import VideoWriter, VideoWriter_fourcc
import os


class Vid_maker:
    def __init__(self, game_name, height, width, fps=120):
        self.game_name = game_name
        self.height = height
        self.width = width
        self.fps = fps
        self.reset()

    def reset(self):
        timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        self.temp_video_path = '/tmp/{}_{}.mp4'.format(self.game_name, timestamp)
        fourcc = VideoWriter_fourcc(*'mp4v')
        self.video = VideoWriter(self.temp_video_path, fourcc, float(self.fps), (self.width, self.height))

    def add_frame(self, frame):
        frame_bgr = frame[..., ::-1]
        self.video.write(frame_bgr)

    def close(self):
        self.video.release()
        filename = os.path.basename(self.temp_video_path)
        shutil.move(self.temp_video_path, './{}'.format(filename))
        self.video = None
        self.reset()
