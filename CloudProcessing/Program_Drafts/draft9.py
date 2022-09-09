import spectral.io.envi as envi
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from scipy.linalg import norm 
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from sklearn.neighbors import KNeighborsClassifier

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
    cloud_loc_list = []
    non_cloud_loc_list = []
    max_rows = np.shape(annotation_array)[0]
    print(max_rows)
    max_columns = np.shape(annotation_array)[1]
    print(max_columns)
    for i in range(max_rows):
        for j in range(max_columns):
            pixel_value = annotation_array[i, j]
            if pixel_value[2] > 240 and pixel_value[1] < 15 and pixel_value[0] < 15:
                # means that the pixel is red
                cloud_spectrum = array_3d[i, j, :]
                if cloud_spectrum.size > 0:
                    OneD_cloud_spectrum = np.squeeze(cloud_spectrum)
                    cloud_list.append(OneD_cloud_spectrum)
                    # add row column spectrum from array_3d to red array
                    location = i, j
                    cloud_loc_list.append(location)
            if pixel_value[1] > 240 and pixel_value[0] < 15 and pixel_value[2] < 15:
                non_cloud_spectrum = array_3d[i, j, :]
                if non_cloud_spectrum.size > 0:
                    OneD_non_cloud_spectrum = np.squeeze(non_cloud_spectrum)
                    non_cloud_list.append(OneD_non_cloud_spectrum)
                    location = i, j
                    non_cloud_loc_list.append(location)
    return cloud_loc_list, non_cloud_loc_list, cloud_list, non_cloud_list 
   

def read_file(infile):
    infile = infile.strip('\n')
    wavelengths = []
    I = envi.open(infile + ".hdr")
    # close file/?
    array_3d = I.load()
    annotation_array = cv2.imread(infile + "color.png")
    return array_3d, wavelengths, annotation_array

def get_input_output_arrays(cloud_loc_list, non_cloud_loc_list, cloud_list, non_cloud_list):
    cloud_size = len(cloud_list)
    non_cloud_size = len(non_cloud_list)
    cloud_class = cloud_size*['cloud']
    non_cloud_class = non_cloud_size*['non-cloud']
    y = cloud_class + non_cloud_class 
    x = cloud_list + non_cloud_list
    loc_list = cloud_loc_list + non_cloud_loc_list
    return x, y, loc_list

def get_train_test_indices(y_array):
    size = len(y_array)
    indices = np.random.permutation(size)
    xy_train_end = int(size//2)
    print(xy_train_end)
    print("end index")
    xy_train_indices = indices[0:xy_train_end]
    xy_test_indices = indices[xy_train_end:size]
    return xy_train_indices, xy_test_indices

def train_test(x_array, y_array, loc_list, model):
    xy_train_indices, xy_test_indices = get_train_test_indices(y_array)
    x_train = []
    y_train = [] 
    x_test = []
    y_test = []
    loc_test = []
    x_array = np.array(x_array)
    for index in xy_train_indices:
        x_train.append(x_array[index])
        y_train.append(y_array[index])
    np.array(x_train)
    np.array(y_train)
    for index in xy_test_indices:
        x_test.append(x_array[index])
        y_test.append(y_array[index])
        loc_test.append(loc_list[index])
    model.fit(x_train, y_train)
    #predictions = model.predict(x_test)
    #score = accuracy_score(y_test, predictions)
    #print(score)
    np.array(x_test)
    np.array(y_test)
    np.array(loc_test)
    return x_test, loc_test, y_test 

def calculate_reflectance(radiances):
    reflectance_array = []
    for radiance in radiances:
        norm_rad = norm(radiance)
        reflectance = radiance/norm_rad
        reflectance_array.append(reflectance)
    reflectance_array = np.array(reflectance_array)
    return reflectance_array

def make_predictions(x_test, y_test, model):
    predictions = model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    print(score)
    return predictions

def create_classification_maps(predictions, files, annotation_array_files, loc_test_files, indices):
    for i in range(files):
        infile = files[i]
        infile = infile.strip('\n')
        annotation_array = annotation_array_files[i]
        start = indices[i]
        end = indices[i+1]
        prediction_file = predictions[start:end:1]
        loc_test = loc_test_files[start:end:1]
        for count in range(len(prediction_file)):
            row, column = loc_test[count]
            if predictions[count] == 'cloud': 
                annotation_array[row, column] = np.array([255, 0, 0])
                # if program identifies as cloud, turns pixel blue
            if predictions[count] == 'non-cloud':
                annotation_array[row, column] = np.array([0, 255, 255])
                #if program identifies a non-cloud area, turns the pixel yellow
        cv2.imwrite("annotated" + infile + ".png", annotation_array)
    print("End of program")
        
def save_test_values(x_test, y_test, loc_test, annotation_array, x_test_files, y_test_files, loc_test_files, indices, annotation_array_files):
    x_test_files += x_test
    y_test_files += y_test
    loc_test_files += loc_test
    index = [len(x_test)]
    indices += index
    annotation_array_files += [annotation_array]
    return x_test_files, y_test_files, loc_test_files, indices, annotation_array_files

if __name__ == "__main__":
    with open('AERONET_images.txt') as f:
        files = f.readlines()
    model = KNeighborsClassifier(n_neighbors=3)
    x_test_files = []
    np.array(x_test_files)
    y_test_files = []
    np.array(x_test_files)
    loc_test_files = []
    np.array(loc_test_files)
    indices = [0]
    annotation_array_files = [] 
    for infile in files:
        array_3d, wavelengths, annotation_array = read_file(infile)
        cloud_loc_list, non_cloud_loc_list, cloud_list, non_cloud_list = get_cloud_lists(array_3d, annotation_array)
        #import ipdb; ipdb.set_trace()
        radiances, label, loc_list = get_input_output_arrays(cloud_loc_list, non_cloud_loc_list, cloud_list, non_cloud_list)
        reflectance = calculate_reflectance(radiances)
        #graph_one_reflectance_spectrum(wavelengths, reflectance)
        x_test, loc_test, y_test = train_test(reflectance, label, loc_list, model)
        x_test_files, y_test_files, loc_test_files, indices, annotation_array_files = save_test_values(x_test, y_test, loc_test, annotation_array, x_test_files, y_test_files, loc_test_files, indices, annotation_array_files)
    predictions = make_predictions(x_test_files, y_test_files, model)
    create_classification_maps(predictions, files, annotation_array_files, loc_test_files, indices)