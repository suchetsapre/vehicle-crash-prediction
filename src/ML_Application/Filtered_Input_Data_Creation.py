"""
Features per frame: Number of cars, positions of the cars (could instead be the normalized distance between each
car and the center of the video), their relative depths, and velocities (would change frame by frame).
"""

import numpy as np
import math
import pandas as pd
from src.Naive_Crash_Predictor import NCP_Algorithm
import os
import random

scale_factor = 100


def normalize_car_points(centered, x_max, y_max):
    normalized_centered = []
    for i in range(len(centered)):
        normalized_i_centered = []
        for j in range(len(centered[i])):
            curr_x, curr_y = centered[i][j]
            norm_x, norm_y = curr_x / x_max, curr_y / y_max
            normalized_i_centered.append((norm_x, norm_y))
        normalized_centered.append(normalized_i_centered)
    return normalized_centered


def get_velocities(centered):
    """ Centered is a list of the normalized car center points. """

    velocities = []
    for i in range(len(centered)):
        curr_velocities = []

        if i == 0:
            for j in range(len(centered[i])):
                if j >= len(centered[i + 1]):
                    curr_velocities.append(0)
                    continue
                x_i, y_i = centered[i][j]
                x_f, y_f = centered[i + 1][j]
                dist_traveled = math.sqrt((x_f - x_i) ** 2 + (y_f - y_i) ** 2)
                curr_car_j_velocity = scale_factor * dist_traveled
                curr_velocities.append(curr_car_j_velocity)
        elif i == len(centered) - 1:
            for j in range(len(centered[i])):
                x_i, y_i = centered[i - 1][j]
                x_f, y_f = centered[i][j]
                dist_traveled = math.sqrt((x_f - x_i) ** 2 + (y_f - y_i) ** 2)
                curr_car_j_velocity = scale_factor * dist_traveled
                curr_velocities.append(curr_car_j_velocity)
        else:
            for j in range(len(centered[i])):
                x_i, y_i = centered[i - 1][j]
                x_f, y_f = centered[i + 1][j]
                dist_traveled = math.sqrt((x_f - x_i) ** 2 + (y_f - y_i) ** 2)
                curr_car_j_velocity = scale_factor * dist_traveled / 2
                curr_velocities.append(curr_car_j_velocity)
        velocities.append(curr_velocities)

    return velocities


def convert_video_to_data(filename):
    frames = NCP_Algorithm.read_images_from_video(filename)
    output, car_points, centered, depths = NCP_Algorithm.object_tracking(frames)
    x_max, y_max = frames[0].shape[0], frames[0].shape[1]
    normalized_centered = normalize_car_points(centered, x_max, y_max)
    velocities = get_velocities(normalized_centered)
    num_cars = len(depths[0])
    video_data = []

    for i in range(len(frames)):
        frame_data = [num_cars]
        for j in range(len(centered[i])):
            pos_x, pos_y = normalized_centered[i][j]
            depth = depths[i][j]
            velocity = velocities[i][j]
            frame_data.append(pos_x)
            frame_data.append(pos_y)
            frame_data.append(velocity)
            frame_data.append(depth)
        frame_data = np.array(frame_data)
        video_data.append(frame_data)

    video_data = np.array(video_data)

    if video_data.ndim == 1:
        video_data = np.reshape(video_data, (100, 1))

    return video_data


def create_data(sample_pct):
    """
        Train has 200 records and test and 100 records.
        TODO: Videos may have different numbers of cars, so robust padding is critical
            Create both the training and testing datasets at the same time.
            Save your pad_length!
    """
    # sample_pct += 0.03 #accounts for any bad videos

    # POSITIVE SETUP
    df_train = pd.read_csv('positive_training_videos_crash_annotation.csv')
    df_test = pd.read_csv('positive_testing_videos_crash_annotation.csv')

    train_video_names = df_train.iloc[:, 0]
    test_video_names = df_test.iloc[:, 0]

    train_num, test_num = int(round(sample_pct * df_train.shape[0])), int(round(sample_pct * df_test.shape[0]))

    # train_num is the number of positive samples in the training, so the total train_num is 2*train_num

    print('%i training samples and %i testing samples' % (2 * train_num, 2 * test_num))

    train_sampled_indices = random.sample([i for i in range(df_train.shape[0])], train_num)
    test_sampled_indices = random.sample([i for i in range(df_test.shape[0])], test_num)

    train_sampled_video_names = train_video_names[train_sampled_indices]
    test_sampled_video_names = test_video_names[test_sampled_indices]

    # NEGATIVE SETUP
    # Example: './videos/training/negative/'

    folder_path_training = './videos/training/negative/'
    negative_training_videos = []
    for file_name in os.listdir(folder_path_training):
        negative_training_videos.append(folder_path_training + file_name)

    folder_path_testing = './videos/testing/negative/'
    negative_testing_videos = []
    for file_name in os.listdir(folder_path_testing):
        negative_testing_videos.append(folder_path_testing + file_name)

    train_sampled_video_names_negative = random.sample(negative_training_videos, train_num)
    test_sampled_video_names_negative = random.sample(negative_testing_videos, test_num)

    # x-data
    print('Processing Training Videos')

    true_train_num = 2 * train_num
    true_test_num = 2 * test_num
    count = 1

    x_data = []
    y_train_true = []
    y_test_true = []

    # Changing the dataframe format!

    df_train.index = df_train.iloc[:, 0]
    df_train.drop(df_train.columns[[0]], axis=1, inplace=True)

    df_test.index = df_test.iloc[:, 0]
    df_test.drop(df_test.columns[[0]], axis=1, inplace=True)

    for video_name in train_sampled_video_names:
        print('--------Video %i of %i---------' % (count, true_test_num + true_train_num))
        try:
            x_data.append(convert_video_to_data(video_name))
        except Exception as e:
            print('Error Processing Video. Exception: ' + str(e) + ' Skipping Video.')
            true_train_num -= 1
        else:
            print('No Error When Processing Video!')
            crash_frame = df_train.loc[video_name][0]
            temp = np.zeros(100)
            temp[crash_frame - 24:crash_frame + 1] = np.ones(25)
            y_train_true.append(temp)
            count += 1
        print(len(x_data))
        print(len(y_train_true))

    for video_name in train_sampled_video_names_negative:
        print('--------Video %i of %i---------' % (count, true_test_num + true_train_num))
        try:
            x_data.append(convert_video_to_data(video_name))
        except Exception as e:
            print('Error Processing Video. Exception: ' + str(e) + ' Skipping Video.')
            true_train_num -= 1
        else:
            print('No Error When Processing Video!')
            y_train_true.append(np.zeros(100))
            count += 1

    print('Processing Testing Videos')

    for video_name in test_sampled_video_names:
        print('--------Video %i of %i---------' % (count, true_test_num + true_train_num))
        try:
            x_data.append(convert_video_to_data(video_name))
        except Exception as e:
            print('Error Processing Video. Exception: ' + str(e) + ' Skipping Video.')
            true_test_num -= 1
        else:
            print('No Error When Processing Video!')
            crash_frame = df_test.loc[video_name][0]
            temp = np.zeros(100)
            temp[crash_frame - 24:crash_frame + 1] = np.ones(25)
            y_test_true.append(temp)
            count += 1

    for video_name in test_sampled_video_names_negative:
        print('--------Video %i of %i---------' % (count, true_test_num + true_train_num))
        try:
            x_data.append(convert_video_to_data(video_name))
        except Exception as e:
            print('Error Processing Video. Exception: ' + str(e) + ' Skipping Video.')
            true_test_num -= 1
        else:
            print('No Error When Processing Video!')
            y_test_true.append(np.zeros(100))
            count += 1

    pad_len = 0
    for vid_data in x_data:
        for i in range(vid_data.shape[0]):
            curr_frame_data_len = vid_data[i].shape[0]
            if curr_frame_data_len > pad_len:
                pad_len = curr_frame_data_len

    print('Pad Length: %i' % pad_len)

    x_data_padded = []
    for vid_data in x_data:
        print(vid_data.shape)
        pad_vid = np.pad(vid_data, ((0, 0), (0, pad_len - vid_data.shape[1])), 'constant')
        x_data_padded.append(pad_vid)

    x_train_true = x_data_padded[0:true_train_num]
    x_test_true = x_data_padded[true_train_num:]

    y_train_true, y_test_true = np.array(y_train_true), np.array(y_test_true)
    x_train_true, x_test_true = np.array(x_train_true), np.array(x_test_true)

    print(x_train_true.shape)
    print(x_test_true.shape)

    print('Number of Final Training Samples %i' % true_train_num)
    print('Number of Final Testing Samples %i' % true_test_num)

    np.save('X_train_100pct.npy', x_train_true)
    np.save('X_test_100pct.npy', x_test_true)
    np.save('y_train_100pct.npy', y_train_true)
    np.save('y_test_100pct.npy', y_test_true)

    print('Data Creation Complete!')


def create_y_data():
    y_train = pd.read_csv('positive_training_videos_crash_annotation.csv')
    y_test = pd.read_csv('positive_testing_videos_crash_annotation.csv')
    y_train, y_test = y_train.iloc[:, 1].values, y_test.iloc[:, 1].values

    y_train_true = []
    for x in y_train:
        temp = np.zeros(100)
        temp[x - 24:x + 1] = np.ones(25)
        y_train_true.append(temp)
    y_train_true = np.array(y_train_true)

    y_test_true = []
    for x in y_test:
        temp = np.zeros(100)
        temp[x - 24:x + 1] = np.ones(25)
        y_test_true.append(temp)
    y_test_true = np.array(y_test_true)

    print(y_train_true)
    print(y_test_true)

    np.save('y_train.npy', y_train_true)
    np.save('y_test.npy', y_test_true)
