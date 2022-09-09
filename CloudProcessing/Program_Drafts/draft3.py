import spectral.io.envi as envi
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from scipy.linalg import norm 
import matplotlib.pyplot as plt

def graph_one_reflectance_spectrum(wavelength, reflectance):
    x = wavelength
    y = reflectance[0]

# setting the axes at the center
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['left'].set_position('center')
    axis.spines['bottom'].set_position('zero')
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    plt.plot(x,y, 'b')
    plt.show()


def get_cloud_lists(array_3d, annotation_array):
    cloud_list = []
    non_cloud_list = []
    cloud_dict = {}
    non_cloud_dict = {}
    # more efficient
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
                cloud_dict[tuple(OneD_cloud_spectrum)] = i, j
            if pixel_value[1] > 240 and pixel_value[0] < 15 and pixel_value[2] < 15:
                non_cloud_spectrum = array_3d[i, j, :]
                OneD_non_cloud_spectrum = np.squeeze(non_cloud_spectrum)
                non_cloud_list.append(OneD_non_cloud_spectrum)
                non_cloud_dict[tuple(OneD_non_cloud_spectrum)] = i, j
    return cloud_dict, non_cloud_dict, cloud_list, non_cloud_list 
   

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
    annotation_array = cv2.imread(infile + "color.png")
    return array_3d, wavelengths, annotation_array

def get_input_output_arrays(cloud_dict, non_cloud_dict, cloud_list, non_cloud_list):
    cloud_size = len(cloud_list)
    non_cloud_size = len(non_cloud_list)
    cloud_class = cloud_size*['cloud']
    non_cloud_class = non_cloud_size*['non-cloud']
    y = cloud_class + non_cloud_class 
    y_array = np.array(y)
    x = cloud_list + non_cloud_list
    cloud_dict.update(non_cloud_dict)
    x_array = np.array(x)
    return x_array, y_array, cloud_dict

def train_test(x_array, y_array):
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.5)
    model = LogisticRegression()
    print(x_train)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    print(score)
    return x_test, predictions

def calculate_reflectance(radiances, annotated_dict):
    reflectance_array = []
    count = 0
    for radiance in radiances:
        count += 1
        print(count)
        pixel_location = annotated_dict[tuple(radiance)]
        del annotated_dict[tuple(radiance)]
        norm_rad = norm(radiance)
        reflectance = radiance/norm_rad
        annotated_dict[tuple(reflectance)] = pixel_location
        reflectance_array.append(reflectance)
    reflectance_array = np.array(reflectance_array)
    return reflectance_array, annotated_dict

def create_classification_map(x_test, predictions, annotation_array, annotated_dict):
    for count, reflectance in enumerate(x_test):
        row, column = annotated_dict[reflectance]
        if predictions[count] == 'cloud': 
            annotation_array[row, column] = 255, 0, 0 
            # if program identifies as cloud, turns pixel blue
        if predictions[count] == 'non-cloud':
            annotation_array[row, column] = 255, 255, 0
            #if program identifies a non-cloud area, turns the pixel yellow
    # displays the new image
    cv2.imshow("annotated_image", annotation_array)
    cv2.imwrite("annotated_image.png", annotation_array)
    print("End of program")
        

if __name__ == "__main__":
    with open('AERONET_images.txt') as f:
        files = f.readlines()
    for infile in files:
        array_3d, wavelengths, annotation_array = read_file(infile)
        cloud_dict, non_cloud_dict, cloud_list, non_cloud_list = get_cloud_lists(array_3d, annotation_array)
        # save model in between? for different files...fix for multiple files
        radiances, label, annotated_dict = get_input_output_arrays(cloud_dict, non_cloud_dict, cloud_list, non_cloud_list)
        reflectance, annotated_dict = calculate_reflectance(radiances, annotated_dict)
        graph_one_reflectance_spectrum(wavelengths, reflectance)
        x_test, predictions = train_test(reflectance, label)
        create_classification_map(x_test, predictions, annotation_array, annotated_dict)