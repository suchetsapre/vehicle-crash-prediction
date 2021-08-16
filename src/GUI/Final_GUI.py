""" Running this file will allow for experimentation with my GUI """

from __future__ import print_function

import datetime
import math
import os
from tkinter import *

import PIL.Image
import PIL.ImageTk
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pytube

from src.ML_Application import Filtered_Input_Data_Creation
from src.Naive_Crash_Predictor import NCP_Algorithm

# This path will vary based on where the dataset is stored on your local machine.
path_to_dashcam_video_dataset = "../../../Crash_Detection_Project/"

path_to_youtube_vids = './Sample_Generated_YouTube_Videos/'
path_to_generated_prediction_vids = './Sample_Generated_Prediction_Videos/'


class CreateToolTip(object):
    """
    Author: crxguy52
    Date: March 25, 2016

    Create a tooltip for a given widget
    """

    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # miliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tw = Toplevel(self.widget)  # Remove "tk"

        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background="#ffffff", relief='solid', borderwidth=1,
                      wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


class FrameExtractor():
    '''
    Author: erykml
    Link: https://gist.github.com/erykml/6a1fe38763664567e6052e78e047ebb5

    Class used for extracting frames from a video file.
    '''

    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        ''' Prints the duration of the given video '''
        duration = self.n_frames / self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():
            success, image = self.vid_cap.read()

            if not success:
                break

            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1

            frame_cnt += 1

        self.vid_cap.release()
        cv2.destroyAllWindows()

    def extract_first_frame(self, img_name, dest_path=None, img_ext='.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        success, image = self.vid_cap.read()

        if not success: return 0

        img_path = os.path.join(dest_path, ''.join([img_name, '_', str(0), img_ext]))
        cv2.imwrite(img_path, image)

        self.vid_cap.release()
        cv2.destroyAllWindows()


def main():
    global frameRate
    global depthDifferential
    global distanceDifferential
    global lookAhead
    global youtubeFile
    global priority

    youtubeFile = None
    priority = 0  # 0 : drop down; 1 : youtube
    img_size = (300, 175)

    def show_video(cap, wait_time):
        if not cap.isOpened():
            print("Error opening video  file")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def clicked_sim():
        '''
            Video 1: output36.avi
            Video 2: output31.avi
            Video 3: output37.avi
        '''
        header_label1.config(text="Processing")
        video_num = variable.get()
        video_name = ""
        if video_num == "Video 1":
            video_name = "OutputVideos/Demo_Video1.avi"
        elif video_num == "Video 2":
            video_name = "OutputVideos/Demo_Video2.avi"
        elif video_num == "Video 3":
            video_name = "OutputVideos/Demo_Video3.avi"
        cap = cv2.VideoCapture(video_name)

        global frameRate

        wait_time = max(int(1000 / 30), 1)
        show_video(cap, wait_time)
        header_label1.config(text="Video Display")

        print("DONE")

    def clicked_sim2():
        header_label1.config(text="Processing")
        start_frame = 0
        end_frame = 100
        video_name = ""
        video_num = variable.get()

        global priority
        global youtubeFile

        if youtubeFile is not None and priority == 1:
            start_frame = -1
            video_name = path_to_youtube_vids + youtubeFile
        elif video_num == "Video 1":  # video 74
            video_name = path_to_dashcam_video_dataset + 'videos/training/positive/000074.mp4'
            start_frame = 55
            end_frame = 100
        elif video_num == "Video 2":  # video 30
            video_name = path_to_dashcam_video_dataset + 'videos/training/positive/000030.mp4'
            start_frame = 27
            end_frame = 40
        elif video_num == "Video 3":  # video 78
            video_name = path_to_dashcam_video_dataset + 'videos/training/positive/000078.mp4'
            start_frame = 39
            end_frame = 100

        global frameRate
        global depthDifferential
        global distanceDifferential
        global lookAhead

        if priority == 0:
            vid_tag = video_num + '_Depth_' + str(depthDifferential) + '_Distance_' + str(
                distanceDifferential) + '_LookAhead_' + str(lookAhead)
        else:
            vid_tag = youtubeFile[:-4] + '_Depth_' + str(depthDifferential) + '_Distance_' + str(
                distanceDifferential) + '_LookAhead_' + str(lookAhead)

        output_file = path_to_generated_prediction_vids + vid_tag + '.avi'
        NCP_Algorithm.process_predict_output(start_frame, end_frame, video_name, frameRate, depthDifferential,
                                             distanceDifferential, lookAhead, output_file)
        cap = cv2.VideoCapture(output_file)

        wait_time = max(int(1000 / frameRate), 1)

        show_video(cap, wait_time)

        print('DONE')

    def clicked_logo():
        video_url = youtube_entry.get()
        vid = pytube.YouTube(video_url)
        compressed_video_title = vid.title
        cap = cv2.VideoCapture(path_to_youtube_vids + compressed_video_title + ".mp4")

        global frameRate

        wait_time = max(int(1000 / frameRate), 1)

        show_video(cap, wait_time)

    def clicked_yt():
        """
        Potential YouTube Crash Videos
        1. https://www.youtube.com/watch?v=S3QgwUyajys
        2. https://www.youtube.com/watch?v=M3EtizAg9Z4
        3. https://www.youtube.com/watch?v=ybpXPfvZG1Y
        4. https://www.youtube.com/watch?v=jS7TPAO0c7g
        5. https://www.youtube.com/watch?v=5DEdR5lqnDE
        6. https://www.youtube.com/watch?v=5G4-LjIdRL0
        7. https://www.youtube.com/watch?v=ybpXPfvZG1Y
        """

        youtube_label.config(text="Downloading...")

        video_url = youtube_entry.get()

        vid_stream = pytube.YouTube(video_url).streams
        mp4_stream = vid_stream.filter(file_extension="mp4")
        mp4_stream.first().download('./Sample_Generated_YouTube_Videos/')

        youtube_label.config(text="Download Complete! Enter Another YT Link Here")

        vid = pytube.YouTube(video_url)
        compressed_video_title = vid.title

        fe = FrameExtractor(path_to_youtube_vids + compressed_video_title + '.mp4')

        fe.extract_first_frame(compressed_video_title, path_to_youtube_vids)

        im = PIL.Image.open(path_to_youtube_vids + compressed_video_title + '_0' + '.jpg')
        im = im.resize(img_size)
        ph = PIL.ImageTk.PhotoImage(im)
        image_label = Label(window, image=ph)
        image_label.image = ph
        image_label.grid(column=1, row=10)

        global youtubeFile
        global priority

        youtubeFile = compressed_video_title + '.mp4'
        priority = 1

    def get_depth_differential(*args):
        """ Obtains the value from the depth differential scale. """
        global depthDifferential
        depthDifferential = scale1.get()

    def get_distance_differential(*args):
        """ Obtains the value from the distance differential scale. """
        global distanceDifferential
        distanceDifferential = scale2.get()

    def get_frame_rate(*args):
        """ Obtains the value from the frame rate scale. """
        global frameRate
        frameRate = scale3.get()

    def get_look_ahead(*args):
        """ Obtains the value from the look ahead scale. """
        global lookAhead
        lookAhead = scale4.get()

    ''' ML CODE BELOW '''

    def read_images_from_video(filename):
        """ Returns the frames given a video filename. """
        vid_cap = cv2.VideoCapture(filename)
        success, image = vid_cap.read()
        count = 0
        frames = []
        while success:
            frames.append(np.array(PIL.Image.fromarray(image)))
            success, image = vid_cap.read()
            count += 1
        return frames

    def run_ml_model_filtered():
        model = keras.models.load_model(
            '../ML_Models/filtered_input_model_lossoptimization.h5')  # FirstCondensedModelTest.h5 corresponds with 1pct
        vid_num = int(ml_video_number.get())
        vid_num_reformatted = '{:06}'.format(vid_num)
        vid_filepath = 'TEMP'

        if 620 >= vid_num >= 456:
            vid_filepath = path_to_dashcam_video_dataset + 'videos/testing/positive/' + vid_num_reformatted + '.mp4'
        if 1130 >= vid_num >= 830:
            vid_filepath = path_to_dashcam_video_dataset + 'videos/testing/negative/' + vid_num_reformatted + '.mp4'

        x_data = [Filtered_Input_Data_Creation.convert_video_to_data(vid_filepath)]
        x_data_padded = []
        pad_len = 69  # 37 for 1pct datafile, 65 for 10pct datafile, 69 for 100pct datafile
        for vid_data in x_data:
            pad_vid = np.pad(vid_data, ((0, 0), (0, pad_len - vid_data.shape[1])), 'constant')
            x_data_padded.append(pad_vid)

        x_test = np.array(x_data_padded)
        pred = model.predict(x_test)

        plt.plot(pred[0])
        plt.xlabel('Frame Number')
        plt.ylabel('Probability of Crash Within Next 20 Frames')
        plt.savefig('GUI_ScreenShots/ML_Plot_Filtered_Video_' + str(vid_num) + '.png')
        plt.clf()

        ml_im = PIL.Image.open('GUI_ScreenShots/ML_Plot_Filtered_Video_' + str(vid_num) + '.png')
        ml_im = ml_im.resize((int(img_size[0] * 1.3), int(img_size[1] * 1.5)))
        ml_ph = PIL.ImageTk.PhotoImage(ml_im)
        ml_image_label = Label(window, image=ml_ph)
        ml_image_label.image = ml_ph
        ml_image_label.grid(column=6, row=10)

    def run_ml_model_images():
        ''' Given a .h5 file and a video number from the text box, output the probability vs. frame graph of the ML crash prediction model. '''
        model = keras.models.load_model('my_model2.h5')
        vid_num = int(ml_video_number.get())
        vid_num_reformatted = '{:06}'.format(vid_num)
        vid_filepath = 'TEMP'

        if 620 >= vid_num >= 456:
            vid_filepath = path_to_dashcam_video_dataset + 'videos/testing/positive/' + vid_num_reformatted + '.mp4'
        if 1130 >= vid_num >= 830:
            vid_filepath = path_to_dashcam_video_dataset + 'videos/testing/negative/' + vid_num_reformatted + '.mp4'

        x_test = read_images_from_video(vid_filepath)

        for i in range(len(x_test)):
            x_test[i] = cv2.resize(x_test[i], (178, 100), interpolation=cv2.INTER_AREA)
            x_test[i] = cv2.cvtColor(x_test[i], cv2.COLOR_RGB2GRAY)

        x_test = np.reshape(x_test, (100, 100, 178, 1))
        pred = model.predict(x_test)

        plt.plot(pred)
        plt.xlabel('Frame Number')
        plt.ylabel('Probability of Crash Within Next 20 Frames')
        plt.savefig('GUI_ScreenShots/ML_Plot_Images_Video_' + str(vid_num) + '.png')
        plt.clf()

        ml_im = PIL.Image.open('GUI_ScreenShots/ML_Plot_Images_Video_' + str(vid_num) + '.png')
        ml_im = ml_im.resize((int(img_size[0] * 1.3), int(img_size[1] * 1.5)))
        ml_ph = PIL.ImageTk.PhotoImage(ml_im)
        ml_image_label = Label(window, image=ml_ph)
        ml_image_label.image = ml_ph
        ml_image_label.grid(column=6, row=10)

    window = Tk()
    window.title("Car Crash Prediction UI")
    window.geometry("1400x780")

    title_label = Label(window, text="Linear Approximation Approach", font='Helvetica 24 bold')
    title_label.grid(column=1, row=3)

    youtube_label = Label(window, text="Enter Youtube Link Here", font='Helvetica 16')
    youtube_label.grid(column=2, row=1)

    youtube_entry = Entry(window)
    youtube_entry.grid(column=2, row=2)

    youtube_button = Button(window, text="Get Video", bg="orange", fg="red", command=clicked_yt)
    youtube_button.grid(column=3, row=2)
    youtube_button_ttp = CreateToolTip(youtube_button, "Click this button to fetch the YouTube video from the internet")

    im_1 = PIL.Image.open('GUI_ScreenShots/YTLogo.png')
    im_1 = im_1.resize((75, 50))
    ph_1 = PIL.ImageTk.PhotoImage(im_1)
    logo_button = Button(window, image=ph_1, bg="orange", fg="red", command=clicked_logo)
    logo_button.grid(column=3, row=1)
    logo_button_ttp = CreateToolTip(logo_button, "Click this button to play the YouTube video on screen")

    run_button = Button(window, text="Run Baseline Crash Prediction", bg="orange", fg="red", command=clicked_sim)
    run_button.grid(column=0, row=11)

    run_button2 = Button(window, text="Run User-Parameter Crash Prediction", bg="orange", fg="red",
                         command=clicked_sim2)
    run_button2.grid(column=2, row=11)

    run_button_ttp = CreateToolTip(run_button, "Click this button in order to run the crash prediction algorithm")
    run_button2_ttp = CreateToolTip(run_button2,
                                    "Click this button in order to run the crash prediction algorithm using USER selected parameters")

    OPTIONS = [
        "Video 1",
        "Video 2",
        "Video 3"
    ]  # etc

    variable = StringVar(window)
    variable.set(OPTIONS[0])  # default value

    drop_down_label = Label(window, text="Select Video")
    drop_down_label.grid(column=0, row=0)

    dropdown = OptionMenu(window, variable, *OPTIONS)
    dropdown.grid(column=0, row=0)
    dropdown_ttp = CreateToolTip(dropdown, "Use this dropdown to change the test video")

    header_label1 = Label(window, text="Video Display", font='Helvetica 18')
    header_label1.grid(column=1, row=9)

    im = PIL.Image.open('GUI_ScreenShots/Thumbnail_Video1.jpg')
    im = im.resize(img_size)
    ph = PIL.ImageTk.PhotoImage(im)
    image_label = Label(window, image=ph)
    image_label.grid(column=1, row=10)

    def callback(*args):
        global priority
        priority = 0

        print(variable.get())
        if variable.get() == "Video 1":
            im = PIL.Image.open('GUI_ScreenShots/Thumbnail_Video1.jpg')
            im = im.resize(img_size)
            ph = PIL.ImageTk.PhotoImage(im)
            image_label = Label(window, image=ph)
            image_label.image = ph
            image_label.grid(column=1, row=10)
        elif variable.get() == "Video 2":
            im = PIL.Image.open('GUI_ScreenShots/Thumbnail_Video2.jpg')
            im = im.resize(img_size)
            ph = PIL.ImageTk.PhotoImage(im)
            image_label = Label(window, image=ph)
            image_label.image = ph
            image_label.grid(column=1, row=10)
        elif variable.get() == "Video 3":
            im = PIL.Image.open('GUI_ScreenShots/Thumbnail_Video3.jpg')
            im = im.resize(img_size)
            ph = PIL.ImageTk.PhotoImage(im)
            image_label = Label(window, image=ph)
            image_label.image = ph
            image_label.grid(column=1, row=10)

    variable.trace('w', callback)

    header_label2 = Label(window, text="Parameters", font='Helvetica 18')
    header_label2.grid(column=1, row=4)
    header_label2_ttp = CreateToolTip(header_label2,
                                      "Use these scale parameters to adjust the algorithm and video playback")

    scale_label1 = Label(window, text="Depth Differential")
    scale_label1.grid(column=0, row=5)
    scale1 = Scale(window, from_=1, to=100, command=get_depth_differential)
    scale1.grid(column=0, row=6)

    scale_label2 = Label(window, text="Distance Differential")
    scale_label2.grid(column=1, row=5)
    scale2 = Scale(window, from_=1, to=100, command=get_distance_differential)
    scale2.grid(column=1, row=6)

    scale_label3 = Label(window, text="Frame Rate")
    scale_label3.grid(column=2, row=5)
    scale3 = Scale(window, from_=1, to=250, command=get_frame_rate)
    scale3.grid(column=2, row=6)

    scale_label4 = Label(window, text="Look Ahead")
    scale_label4.grid(column=3, row=5)
    scale4 = Scale(window, from_=1, to=20, command=get_look_ahead)
    scale4.grid(column=3, row=6)

    ml_label = Label(window, text="Machine Learning Application", font='Helvetica 24 bold')
    ml_label.grid(column=6, row=3)

    ml_label2 = Label(window, text="Enter Video Number Here. \n Positive (456-620); Negative (830-1130).",
                      font='Helvetica 18')
    ml_label2.grid(column=6, row=5)
    ml_label2_ttp = CreateToolTip(ml_label2, "Select the testing video here")

    ml_video_number = Entry(window)
    ml_video_number.grid(column=6, row=6)

    ml_run_button = Button(window, text="Run ML Algorithm", bg="orange", fg="red", command=run_ml_model_filtered)
    ml_run_button.grid(column=6, row=7)
    ml_run_button_ttp = CreateToolTip(ml_run_button, "Use this button to run the ML algorithm on the selected video")

    ml_graph_label = Label(window, text="Output Display", font='Helvetica 18')
    ml_graph_label.grid(column=6, row=8)

    ml_im = PIL.Image.open('GUI_ScreenShots/Thumbnail_ML_Graph.png')
    ml_im = ml_im.resize(img_size)
    ml_ph = PIL.ImageTk.PhotoImage(ml_im)
    ml_image_label = Label(window, image=ml_ph)
    ml_image_label.grid(column=6, row=10)

    window.mainloop()


if __name__ == "__main__":
    main()
