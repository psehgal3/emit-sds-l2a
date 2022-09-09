import spectral.io.envi as envi
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
import csv 
from math import pi, cos
import matplotlib.pyplot as plt
from scipy.linalg import norm 

def graph_one_reflectance_spectrum(wavelength, reflectance):
    x = wavelength
    y = reflectance[0]

# setting the axes at the center
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot the function
    plt.plot(x,y, 'r')
    # show the plot
    plt.show()


def get_cloud_lists(array_3d, annotation_array):
    cloud_list = []
    non_cloud_list = []
    max_rows = np.shape(annotation_array)[0]
    max_columns = np.shape(annotation_array)[1]
    for i in range(max_rows):
        for j in range(max_columns):
                # more efficient method?
            pixel_value = annotation_array[i, j]
            if pixel_value[2] > 240 and pixel_value[1] < 15 and pixel_value[0] < 15:
                # means that the pixel is red
                cloud_spectrum = array_3d[i, j, :]
                OneD_cloud_spectrum = np.squeeze(cloud_spectrum)
                cloud_list.append(OneD_cloud_spectrum)
                # add row column spectrum from array_3d to red array
            if pixel_value[1] > 240 and pixel_value[0] < 15 and pixel_value[2] < 15:
                non_cloud_spectrum = array_3d[i, j, :]
                OneD_non_cloud_spectrum = np.squeeze(non_cloud_spectrum)
                non_cloud_list.append(OneD_non_cloud_spectrum)
    return cloud_list, non_cloud_list
   

def read_file(infile):
    wavelengths = []
    with open(infile + ".txt") as f:
        line = f.readline()
        while line:
            wavelength = str(round(float(line.split()[0]), 6))
            wavelengths.append(wavelength)
            line = f.readline()
    I = envi.open(infile + ".hdr")
    # close file/?
    array_3d = I.load()
    # stop from changing into float
    annotation_array = cv2.imread(infile + "color.png")
    cloud_list, non_cloud_list = get_cloud_lists(array_3d, annotation_array)
    with open('cloudfiles2.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            file_name = row[0] + "_rdn_v2p9_img"
            if (file_name == infile):
                elevation = row[1] 
                break
    return cloud_list, non_cloud_list, wavelengths, elevation

def get_input_output_arrays(cloud_list, non_cloud_list):
    cloud_size = len(cloud_list)
    non_cloud_size = len(non_cloud_list)
    cloud_class = cloud_size*['cloud']
    non_cloud_class = non_cloud_size*['non-cloud']
    y = cloud_class + non_cloud_class 
    y_array = np.array(y)
    x = cloud_list + non_cloud_list
    x_array = np.array(x)
    return x_array, y_array

def train_test(x_array, y_array):
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.5)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    print(score)

def calculate_TOA_reflectance(radiances):
    reflectance_array = []
    for radiance in radiances:
        #more efficient way?
        # converts radiance to the correct units
        # radiance = radiance /100
        reflectance = radiance / norm(radiance)
        #adjust units so they cancel out
        # reflectance = reflectance/10
        reflectance_array.append(reflectance)
    reflectance = np.array(reflectance_array)
    return reflectance

if __name__ == "__main__":
    with open('AERONET_images.txt') as f:
        files = f.readlines()
    for infile in files:
        cloud_list, non_cloud_list, wavelengths, elevation = read_file(infile)
        # save model in between? for different files...fix for multiple files
        radiance, label = get_input_output_arrays(cloud_list, non_cloud_list)
        reflectance = calculate_TOA_reflectance(radiance)
        graph_one_reflectance_spectrum(wavelengths, reflectance)
        train_test(reflectance, label)