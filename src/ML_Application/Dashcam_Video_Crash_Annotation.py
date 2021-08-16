import cv2
import csv
import os


def list_of_lists_to_csv(to_csv, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(to_csv)


def annotate_positive_videos_output_csv(folder_path):
    """ example folder_path: './videos/training/positive/' """
    csv_file = []
    num_entries = 0

    for file_name in os.listdir(folder_path):
        video_path = folder_path + file_name
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        annotation = 0
        print(video_path, num_entries)

        if not cap.isOpened():
            print("Error opening video file")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_num < 50:
                    wait_time = 1
                else:
                    wait_time = 250
                cv2.imshow('Frame', frame)
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    annotation = frame_num
                    break
                frame_num += 1
            else:
                break

        cap.release()
        csv_file.append([video_path, annotation])
        num_entries += 1

        if num_entries > 100:
            break

    csv_file = sorted(csv_file, key=lambda x: x[0])
    list_of_lists_to_csv(csv_file, 'positive_testing_videos_crash_annotation.csv')
