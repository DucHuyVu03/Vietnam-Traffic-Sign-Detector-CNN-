#Made by Huy VuDuc
#Use this file to extract data from images and labels you got from Kraggle 
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

classes = ['Pedestrian Crossing', 'Equal-level Intersection', 'No Entry', 'Right Turn Only', 'Intersection', 'Intersection with Uncontrolled Road', 'Dangerous Turn', 'No Left Turn', 'Bus Stop', 'Roundabout', 'No Stopping and No Parking', 'U-Turn Allowed', 'Lane Allocation', 'No Left Turn for Motorcycles', 'Slow Down', 'No Trucks Allowed', 'Narrow Road on the Right', 'No Passenger Cars and Trucks', 'Height Limit', 'No U-Turn', 'No U-Turn and No Right Turn', 'No Cars Allowed', 'Narrow Road on the Left', 'Uneven Road', 'No Two or Three-wheeled Vehicles', 'Customs Checkpoint', 'Motorcycles Only', 'Obstacle on the Road', 'Children Present', 'Trucks and Containers', 'No Motorcycles Allowed', 'Trucks Only', 'Road with Surveillance Camera', 'No Right Turn', 'Series of Dangerous Turns', 'No Containers Allowed', 'No Left or Right Turn', 'No Straight and Right Turn', 'Intersection with T-Junction', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (80km/h)', 'Speed limit (40km/h)', 'Left Turn', 'Low Clearance', 'Other Danger', 'Go Straight', 'No Parking', 'Containers Only', 'No U-Turn for Cars', 'Level Crossing with Barriers']
desired_x = 128
desired_y = 128
def imageCropping(image,type, x, y , w, h):
    X = image.shape[1]
    Y = image.shape[0]
    x = int(X*float(x))
    y = int(Y*float(y))
    w = int(X*float(w)/2)
    h = int(Y*float(h)/2)
    print(type,x,y,w,h)
    cropped_img = image[y-h:y+h, x-w:x+w, :]
    return cropped_img
#Get files to preprocessing before training
data_folder = Path('VietnamTrafficSigns')
img_folder = data_folder/ 'images'
label_folder = data_folder / 'labels'

data_image = list(img_folder.glob('*.jpg'))
data_labels = list(img_folder.glob('*.txt'))
total_image = len(data_image)
num_of_image = len(data_image)
#Call each file to get each image and labelq
img_th = 0
save_folder_128x128 = data_folder/'data_128x128'
for labels_dir, image_dir in zip(label_folder.iterdir(), img_folder.iterdir()):
    img_th += 1
    total_image -= 1
    print(f"Image number {img_th}")
    image_last_element = image_dir.stem
    with open(labels_dir, 'r') as file:
        image = cv.imread(str(image_dir), cv.IMREAD_COLOR)
        lines = file.readlines()
        total_line = len(lines)
        for line in lines:
            coor = line.strip()
            coor = coor.split(' ')
            processed_img = imageCropping(image, coor[0], coor[1], coor[2], coor[3], coor[4])
            processed_img = cv.resize(processed_img,(desired_x,desired_y))
            cv.imshow("img",processed_img)
            save_path_128x128 = save_folder_128x128/ f"{image_last_element}_{classes[int(coor[0])]}.jpg"

            # print(save_path_128x128)
            # cv.imwrite(str(save_path_128x128), processed_img)

            cv.destroyAllWindows()
        print(f"{total_image} images remain")
        print("========================")
# coor = '46 0.480729 0.330556 0.042708 0.090741'
# coor = coor.split(' ')
# print(coor)
# X = img.shape[1]
# Y = img.shape[0]
# x = int(X*float(coor[1]))
# y = int(Y*float(coor[2]))
# w = int(X*float(coor[3])/2)
# h = int(Y*float(coor[4])/2)
# print(x,y,w,h)
# cropped_img = img[y-h:y+h, x-w:x+w, :]
