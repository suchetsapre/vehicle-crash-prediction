import math
import random

import cv2
from PIL import Image

from src.Imported_Files.darknet import Darknet
from src.Imported_Files.utils import *

global outputNum

path_to_dashcam_video_dataset = "../../../Crash_Detection_Project/"


def plot_boxes_and_labels(img, boxes, class_names, colors):
    img = img.copy()
    if len(colors) == 0:
        colors = [(0, 255, 0) for i in range(len(boxes))]
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        box = boxes[i]
        x1 = int(np.around((box[0] - box[2] / 2.0) * width))
        y1 = int(np.around((box[1] - box[3] / 2.0) * height))
        x2 = int(np.around((box[0] + box[2] / 2.0) * width))
        y2 = int(np.around((box[1] + box[3] / 2.0) * height))
        cls_id = box[6]
        conf_interval = box[4]
        # draw a bounding box rectangle and label on the image
        color = colors[i]  # (0, 255, 0)  # [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = "{}: {:.4f}".format(class_names[cls_id], conf_interval)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    return img


def plot_boxes_and_labels_regular(img, boxes, class_names, colors):
    if len(colors) == 0:
        colors = [(0, 255, 0) for i in range(len(boxes))]
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        box = boxes[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[0] + box[2])
        y2 = int(box[1] + box[3])
        # cls_id = box[6]
        # conf_interval = box[4]
        # draw a bounding box rectangle and label on the image
        color = colors[i]  # (0, 255, 0)  # [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # text = "{}: {:.4f}".format(class_names[cls_id], conf_interval)
        # cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.5, color, 2)
    return img


def detect_objects_and_return_points(img):
    cfg_file = '../Imported_Files/yolov3.cfg'
    weight_file = '../Imported_Files/yolov3.weights'
    names_file = '../coco.names'  # CHANGED FROM coco.names
    m = Darknet(cfg_file)
    m.load_weights(weight_file)
    class_names = load_class_names(names_file)

    nms_thresh = 0.5
    iou_thresh = 0.4

    original_image = img

    resized_image = cv2.resize(original_image, (m.width, m.height))

    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

    # Print the objects found and the confidence level
    # print_objects(boxes, class_names)

    # Plot the image with bounding boxes and corresponding object class labels
    plotted_img = plot_boxes_and_labels(img, boxes, class_names, [])
    # plot_boxes(original_image, boxes, class_names, plot_labels=True)
    return plotted_img, boxes


def read_images_from_video(filename):
    vid_cap = cv2.VideoCapture(filename)
    success, image = vid_cap.read()
    count = 0
    frames = []
    while success:
        frames.append(np.array(Image.fromarray(image)))
        success, image = vid_cap.read()
        count += 1
    return frames


def get_car_points_from_boxes(boxes, width, height):
    points = []
    centered = []
    for i in range(len(boxes)):
        box = boxes[i]

        if box[6] >= 13 or box[6] == 0:
            continue  # To disable people detection from messing up with the video

        x1 = int(np.around((box[0] - box[2] / 2.0) * width))
        y1 = int(np.around((box[1] - box[3] / 2.0) * height))
        x2 = int(np.around((box[0] + box[2] / 2.0) * width))
        y2 = int(np.around((box[1] + box[3] / 2.0) * height))
        curr_points = (x1, y1, x2, y2)
        curr_centered = ((x1 + x2) / 2, (y1 + y2) / 2)
        points.append(curr_points)
        centered.append(curr_centered)

    return points, centered


def process_all_frames(frames):
    images = []
    car_points = []
    centered = []
    boxes_plus = []

    for i in range(len(frames)):
        print(i)
        plotted_img, boxes = detect_objects_and_return_points(frames[i])
        images.append(plotted_img)
        points, cent = get_car_points_from_boxes(boxes, frames[i].shape[1], frames[i].shape[0])
        car_points.append(points)
        centered.append(cent)
        boxes_plus.append(boxes)

    return images, car_points, centered, boxes_plus


def create_video_from_frames(frames, output_file='./Sample_Generated_Prediction_Videos/outputNONAME.avi',
                             scale_factor=1.0):
    global outputNum
    width = int(frames[0].shape[1] * scale_factor)
    height = int(frames[0].shape[0] * scale_factor)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

    for i in range(len(frames)):
        width = int(frames[i].shape[1] * scale_factor)
        height = int(frames[i].shape[0] * scale_factor)
        dim = (width, height)
        frames[i] = cv2.resize(frames[i], dim, interpolation=cv2.INTER_AREA)
        out.write(frames[i])

    out.release()
    cv2.destroyAllWindows()


def find_closest(pair, prev_points):
    """
    TODO: Can be optimized to run in O(n log n)
    """

    min_dist = 9223372036854775807  # Max Int
    min_ind = 0

    for i in range(len(prev_points)):
        # print(prev_points[i])

        if prev_points[i] == 0:
            continue

        x, y, dummy = prev_points[i]
        curr_dist = math.sqrt((x - pair[0]) ** 2 + (y - pair[1]) ** 2)
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_ind = i

    return min_ind, min_dist


def detect_crashes(images, carpoints, centered):
    """ Old crash detection algorithm. """
    images = images.copy()
    updated_car_points = [[] for i in range(len(centered))]
    crash_frames = []
    color = (0, 0, 255)

    for i in range(len(centered)):
        curr_points = centered[i]
        print('i value: ' + str(i))
        if i in [0, 1]:
            for j in range(len(curr_points)):
                curr_x, curr_y = curr_points[j]
                updated_car_points[i].append((curr_x, curr_y, 0))
        else:
            for j in range(len(curr_points)):
                ''' Include case where new car point is not close to any of the other car points. Like taking care of
                    adding new points. '''
                curr_x, curr_y = curr_points[j]
                min_ind, min_dist = find_closest((curr_x, curr_y), updated_car_points[i - 2])
                print('Min Dist, Min Index: %f, %i' % (min_dist, min_ind))
                prev_x, prev_y, old_mag = updated_car_points[i - 2][min_ind]

                # Now compare with prev

                new_mag = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                print('Old: %f -- New: %f' % (old_mag, new_mag))

                '''curr_image = images[i]
                x1, y1, x2, y2 = car_points[i][j]
                text = "{:.4f}; {:.4f}; {:.4f}".format(old_mag, new_mag, min_dist)
                cv2.putText(curr_image, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.41, color, 1)
                images[i] = curr_image'''

                if old_mag != 0 and abs(old_mag - new_mag) > 75:
                    crash_frames.append(i)
                    curr_image = images[i]
                    width = curr_image.shape[1]
                    height = curr_image.shape[0]
                    x1, y1, x2, y2 = carpoints[i][j]
                    cv2.rectangle(curr_image, (x1, y1), (x2, y2), color, 2)
                    images[i] = curr_image
                    # print("CRASH at frame %d" % (i))

                updated_car_points[i][min_ind] = (curr_x, curr_y, new_mag)

        if i == len(centered) - 1:
            continue

        updated_car_points[i + 1] = [0 for k in range(max(len(updated_car_points[i]), len(centered[i + 1])))]
    return crash_frames, images


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def create_tracker_by_name(tracker_type):
    # Create a tracker based on tracker name
    if tracker_type == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def object_tracking(images):
    car_points = []
    centered = []

    output = [0 for i in range(len(images))]
    image_first, car_point_first, center_first, boxes_plus = process_all_frames([images[0]])
    image_first, car_point_first, center_first, boxes_plus = image_first[0], car_point_first[0], center_first[0], \
                                                             boxes_plus[0]
    output[0] = image_first
    car_points.append(car_point_first)  # (x,y), (x,y)
    centered.append(center_first)
    classes = [x[6] for x in boxes_plus if x[6] != 0]
    bboxes = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in car_point_first]
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in
              range(len(bboxes))]
    names_file = '../coco_relevant.names'

    # Specify the tracker type
    tracker_type = "BOOSTING"

    # Create MultiTracker object
    multi_tracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    width = images[0].shape[1]
    height = images[0].shape[0]
    for bbox in bboxes:
        x, y, w, h = bbox
        if x < 0 or x + w >= width:
            continue
        if y < 0 or y + h >= height:
            continue
        multi_tracker.add(create_tracker_by_name(tracker_type), images[0], bbox)

    for i in range(1, len(images)):
        if i % 10 == 0: print('Tracker on frame %i of %i' % (i, len(images) - 1))
        success, boxes = multi_tracker.update(images[i])
        centered.append([(int(x + w / 2), int(y + h / 2)) for x, y, w, h in boxes])
        car_points.append([(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in boxes])
        class_names = load_class_names(names_file)
        output[i] = plot_boxes_and_labels_regular(images[i], boxes, class_names, [])

    depths = depth_assignment(car_points, classes)
    color = (0, 255, 0)
    for i in range(len(depths)):
        curr_img = output[i]
        for j in range(len(depths[i])):
            x1, y1, x2, y2 = car_points[i][j]
            text = "{:.4f}".format(depths[i][j])
            cv2.putText(curr_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        output[i] = curr_img
    return output, car_points, centered, depths


def predict_crashes_with_tracking_visualization(images, carpoints, centered):
    """ This method plots the yellow and blue boxes showing the extrapolated
        car location. The blue boxes represents when two of these paths overlap."""
    images = images.copy()
    lh = 5
    # plot 5 frames forward
    # check size of the cars to indicate depth
    color = (0, 255, 255)
    wh = [(x[2] - x[0], x[3] - x[1]) for x in carpoints[0]]
    for i in range(len(centered)):
        look_ahead = [[(0, 0) for i in range(len(centered[i]))] for j in range(lh + 1)]

        if i in [0]: continue

        dxs = [0 for i in range(len(centered[i]))]
        dys = [0 for i in range(len(centered[i]))]

        for j in range(len(centered[i])):
            dxs[j] = centered[i][j][0] - centered[i - 1][j][0]
            dys[j] = centered[i][j][1] - centered[i - 1][j][1]

        look_ahead[0] = centered[i]
        for j in range(1, lh + 1):
            for k in range(len(centered[i])):
                xc = look_ahead[j - 1][k][0] + dxs[k] * 2
                yc = look_ahead[j - 1][k][1] + dys[k] * 2
                look_ahead[j][k] = (xc, yc)
                # print(look_ahead[j][k])
                curr_image = images[i]
                w, h = wh[k]
                cv2.rectangle(curr_image, (int(xc - w / 2), int(yc - h / 2)), (int(xc + w / 2), int(yc + h / 2)), color,
                              2)
                images[i] = curr_image

        for j in range(1, lh + 1):
            min_cars = (0, 0)
            min_dist = 500000000
            wh_ind = (0, 0)
            for k in range(0, len(look_ahead[j]) - 1):
                for l in range(k + 1, len(look_ahead[j])):
                    x1, y1 = look_ahead[j][k]
                    x2, y2 = look_ahead[j][l]
                    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        min_cars = (x1, y1, x2, y2)
                        wh_ind = (k, l)

            x1, y1, x2, y2 = min_cars
            w1, h1 = wh[wh_ind[0]]
            w2, h2 = wh[wh_ind[1]]
            curr_image = images[i]
            cv2.rectangle(curr_image, (int(x1 - w1 / 2), int(y1 - h1 / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)),
                          (255, 255, 0),
                          2)
            cv2.rectangle(curr_image, (int(x2 - w2 / 2), int(y2 - h2 / 2)), (int(x2 + w2 / 2), int(y2 + h2 / 2)),
                          (255, 255, 0),
                          2)
            images[i] = curr_image

    return images


def depth_assignment(car_points, classes):
    depth = [[0 for j in range(len(car_points[i]))] for i in range(len(car_points))]
    ind = 0

    if 2 in classes:  # 2 is a car
        ind = classes.index(2)
    elif 3 in classes:
        ind = classes.index(3)
    elif 7 in classes:
        ind = classes.index(7)
    else:
        ind = classes.index(random.choice(classes))

    for i in range(len(car_points)):
        # classes_areas = [0 for i in set(classes)]
        # for cls in set(classes):
        ref_x1, ref_y1, ref_x2, ref_y2 = car_points[i][ind]  # switch to ind
        ref_w = abs(ref_x2 - ref_x1)
        ref_h = abs(ref_y2 - ref_y1)
        ref_area = ref_w * ref_h
        # classes_areas[cls] = refArea
        depth[i][ind] = 1

        for j in range(len(car_points[i])):
            # cls = classes[j]
            # refArea = classes_areas[cls]
            x1, y1, x2, y2 = car_points[i][j]
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            curr_area = w * h
            d = ref_area / curr_area
            if classes[j] == 3:
                d *= 0.563851892
            if classes[j] == 7:
                d *= 1.45
            depth[i][j] = d

    return depth


def predict_crashes_with_tracking(images, car_points, centered, depths, depth_differential=15, distance_differential=50,
                                  look_ahead=10):
    """ Current crash prediction method. """
    images = images.copy()
    lh = look_ahead

    # plot 5 frames forward
    # check size of the cars to indicate depth

    color = (0, 255, 255)
    wh = [(x[2] - x[0], x[3] - x[1]) for x in car_points[0]]
    for i in range(len(centered)):
        look_ahead = [[(0, 0) for i in range(len(centered[i]))] for j in range(lh + 1)]

        if i == 0:
            continue

        dxs = [0 for i in range(len(centered[i]))]
        dys = [0 for i in range(len(centered[i]))]

        for j in range(len(centered[i])):
            dxs[j] = centered[i][j][0] - centered[i - 1][j][0]
            dys[j] = centered[i][j][1] - centered[i - 1][j][1]

        look_ahead[0] = centered[i]
        for j in range(1, lh + 1):
            for k in range(len(centered[i])):
                xc = look_ahead[j - 1][k][0] + dxs[k]
                yc = look_ahead[j - 1][k][1] + dys[k]
                look_ahead[j][k] = (xc, yc)

        for j in range(1, lh + 1):
            for k in range(0, len(look_ahead[j]) - 1):
                for l in range(k + 1, len(look_ahead[j])):
                    x1, y1 = look_ahead[j][k]
                    x2, y2 = look_ahead[j][l]
                    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < (images[0].shape[0] / 18) * ((distance_differential + 50) / 100) and abs(
                            math.sqrt(depths[min(i + lh, len(car_points) - 1)][k]) - math.sqrt(
                                    depths[min(i + lh, len(car_points) - 1)][
                                        l])) < depth_differential / 100:  # max(depths[min(i+lh, len(car_points)-1)][
                        # k]/depths[min(i+lh, len(car_points)-1)][l], depths[min(i+lh, len(car_points)-1)][l]/depths[
                        # min(i+lh, len(car_points)-1)][k]) - 1 < 0.1:#9.5:#0.30: #this is the pixel number which I
                        # use for the threshold
                        curr_image = images[i]
                        a1, a2, b1, b2 = car_points[i][k]
                        c1, c2, d1, d2 = car_points[i][l]
                        cv2.rectangle(curr_image, (a1, a2), (b1, b2), (0, 0, 255), 5)
                        cv2.rectangle(curr_image, (c1, c2), (d1, d2), (0, 0, 255), 5)
                        images[i] = curr_image
    return images


def detect_crashes_with_tracking(images, car_points, centered):
    images = images.copy()
    updated_car_points = [[] for i in range(len(centered))]
    crash_frames = []
    color = (0, 0, 255)

    for i in range(len(centered)):
        curr_points = centered[i]
        print('i value: ' + str(i))
        if i in [0]:
            for j in range(len(curr_points)):
                curr_x, curr_y = curr_points[j]
                updated_car_points[i].append((curr_x, curr_y, 0))
        else:
            for j in range(len(curr_points)):
                ''' Include case where new car point is not close to any of the other car points. Like taking care of
                    adding new points. '''
                curr_x, curr_y = curr_points[j]
                prev_x, prev_y, old_mag = updated_car_points[i - 1][j]

                # Now compare with prev

                new_mag = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                print('Old: %f -- New: %f' % (old_mag, new_mag))

                '''curr_image = images[i]
                x1, y1, x2, y2 = car_points[i][j]
                text = "{:.4f}; {:.4f}; {:.4f}".format(old_mag, new_mag, min_dist)
                cv2.putText(curr_image, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.41, color, 1)
                images[i] = curr_image'''

                if old_mag != 0 and abs(old_mag - new_mag) > 25:
                    crash_frames.append(i)
                    curr_image = images[i]
                    width = curr_image.shape[1]
                    height = curr_image.shape[0]
                    x1, y1, x2, y2 = car_points[i][j]
                    cv2.rectangle(curr_image, (x1, y1), (x2, y2), color, 5)
                    images[i] = curr_image
                    # print("CRASH at frame %d" % (i))

                updated_car_points[i][j] = (curr_x, curr_y, new_mag)

        if i == len(centered) - 1:
            continue

        updated_car_points[i + 1] = [0 for k in range(max(len(updated_car_points[i]), len(centered[i + 1])))]

    return crash_frames, images


def full_run():
    global outputNum
    outputNum = 42
    frames = read_images_from_video(path_to_dashcam_video_dataset + 'videos/training/positive/000030.mp4')
    # frames = read_images_from_video('IMG_1404.MOV')
    images, car_points, centered, depths = object_tracking(
        frames)  # output image array is len 99 not 100 because i dont update first frame
    rectified_images = predict_crashes_with_tracking(images, car_points, centered, depths)

    # images, car_points, centered = process_all_frames(frames) crash_frames, rectified_images = detect_crashes(
    # images, car_points, centered) #crash_frames is a list of integers, rectified images contains the frames which
    # plot the crash print('---Crash Frames---') print(crash_frames) create_video_from_frames(rectified_images)
    # crash_frames, rectified_images = detect_crashes_with_tracking(images, car_points, centered) print('Finished
    # Tracking!')
    create_video_from_frames(rectified_images)
    # print('---Crash Frames---')
    # print(crash_frames)


def process_predict_output(start_frame, end_frame, video_file, frame_rate, depth_differential, distance_differential,
                           look_ahead, output_file):
    frames = read_images_from_video(video_file)
    height = frames[0].shape[0]
    scale_factor = 650 / height

    if start_frame == -1:
        start_frame = int(len(frames) * 0.32)
        end_frame = int(len(frames) * 0.45)
        frames = frames[start_frame:end_frame]
    else:
        frames = frames[start_frame: end_frame]

    images, car_points, centered, depths = object_tracking(frames)
    rectified_images = predict_crashes_with_tracking(images, car_points, centered, depths, depth_differential,
                                                     distance_differential, look_ahead)
    create_video_from_frames(rectified_images, output_file, scale_factor)

# to incorporate ML
# https://github.com/rwk506/CrashCatcher

# full_run()
