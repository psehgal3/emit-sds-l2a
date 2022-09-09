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
    count2 = 0
    max_rows = np.shape(annotation_array)[0]
    max_columns = np.shape(annotation_array)[1]
    print(np.shape(array_3d))
    print(max_rows)
    print("max_rows")
    print(max_columns)
    array_3d = np.reshape(array_3d,(max_rows, max_columns, 285))
    print(np.shape(array_3d))
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
   

def read_file(infile, count):
    infile = infile.strip('\n')
    wavelengths = []
    print(infile)
    #if count == 1:
    #   with open(infile + ".txt") as f:
     #     line = f.readline()
      #    while line:
       #       wavelength = str(round(float(line.split()[0]), 6))
        #      wavelengths.append(wavelength)
         #     line = f.readline()
    I = envi.open(infile + ".hdr")
    # close file/?
    array_3d = I.load()
    if array_3d.shape[2] > 285:
      array_3d = array_3d[:,:,:285]
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

def train_test(x_array, y_array, loc_list):
    xy_train_indices, xy_test_indices = get_train_test_indices(y_array)
    x_train = []
    y_train = [] 
    x_test = []
    y_test = []
    loc_test = []
    x_train_capped = []
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
    #print(x_test[0])
    #count1 = 0
    #more efficient way to do this?
    #why only 285 radiance values?
    #count5 = 0
    #for element in y_train:
      #if element == "non-cloud":
         #count5 += 1
    #print(count5)
    #print("non_cloud_count")
    #for element in x_train:
      #element_capped = element[0:284]
      #if count1 == 0:
         #print(len(new_element))
         #print("train size")
         #print(new_element)
         #print(element)
         #print(len(element))
      #x_train_capped.append(element_capped)
      #count1 += 1
    #model.partial_fit(x_train_capped, y_train, classes=np.unique(y))
    #predictions = model.predict(x_test)
    #score = accuracy_score(y_test, predictions)
    #print(score)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    #loc_test = np.array(loc_test)
    return x_train, y_train, x_test, loc_test, y_test 

def calculate_reflectance(radiances):
    reflectance_array = []
    for radiance in radiances:
        norm_rad = norm(radiance)
        reflectance = radiance/norm_rad
        reflectance_array.append(reflectance)
    reflectance_array = np.array(reflectance_array)
    return reflectance_array

def make_predictions(x_test, y_test, x_train, y_train):
    print(len(x_train))
    print(len(y_train))
    print("length")
    model = KNeighborsClassifier(n_neighbors=3)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model.fit(x_train, y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    predictions = model.predict(x_test)
    print(len(predictions))
    print("predictions")
    score = accuracy_score(y_test, predictions)
    print(score)
    return predictions

def create_classification_maps(predictions, files, annotation_array_files, loc_test_files, indices):
    for i in range(len(files)):
        infile = files[i]
        infile = infile.strip('\n')
        #print(annotation_array_files)
        #print("annotation_array_files")
        annotation_array = annotation_array_files[i]
        #print(annotation_array)
        start = indices[i]
        print(start)
        end = indices[i+1]
        print(end)
        print("end")
        prediction_file = predictions[start:end]
        loc_test = loc_test_files[start:end]
        for count in range(len(prediction_file)):
            row, column = loc_test[count]
            if prediction_file[count] == 'cloud': 
                annotation_array[row, column] = np.array([255, 0, 0])
                # if program identifies as cloud, turns pixel blue
            if prediction_file[count] == 'non-cloud':
                annotation_array[row, column] = np.array([0, 255, 255])
                #if program identifies a non-cloud area, turns the pixel yellow
        cv2.imwrite("annotated" + infile + ".png", annotation_array)
    print("End of program")
        
def save_test_values(x_train, y_train, x_test, y_test, loc_test, annotation_array, x_test_files, y_test_files, loc_test_files, indices, annotation_array_files, x_train_files, y_train_files, tot_sum):
    #more efficient method?
    #for test_element in x_test:
      #new_test_element = test_element[0:285]
      #x_test_files.append(new_test_element)
    #for train_element in x_train:
      #new_train_element = train_element[0:285]
      #x_train_files.append(new_train_element) 
    x_test_files += x_test
    x_train_files += x_train
    y_test_files += y_test
    y_train_files += y_train
    loc_test_files += loc_test
    index = len(x_test)
    tot_sum += index
    print(index)
    indices += [tot_sum]
    print(indices)
    print("indices")
    annotation_array_files += [annotation_array]
    return x_train_files, y_train_files, x_test_files, y_test_files, loc_test_files, indices, annotation_array_files, tot_sum

if __name__ == "__main__":
    with open('AERONET_images.txt') as f:
        files = f.readlines()
    x_test_files = []
    #x_test_files = np.array(x_test_files)
    #warm_start or prefit method?????
    y_test_files = []
    np.array(y_test_files)
    loc_test_files = []
    np.array(loc_test_files)
    indices = [0]
    annotation_array_files = []
    x_train_files = []
    y_train_files = []
    tot_sum = 0
    count = 0
    for infile in files:
        count += 1
        array_3d, wavelengths, annotation_array = read_file(infile, count)
        #if len(wavelengths) != 0:
           #total_wavelength = len(wavelengths)  
        cloud_loc_list, non_cloud_loc_list, cloud_list, non_cloud_list = get_cloud_lists(array_3d, annotation_array)
        #import ipdb; ipdb.set_trace()
        radiances, label, loc_list = get_input_output_arrays(cloud_loc_list, non_cloud_loc_list, cloud_list, non_cloud_list)
        reflectance = calculate_reflectance(radiances)
        #graph_one_reflectance_spectrum(wavelengths, reflectance)
        x_train, y_train, x_test, loc_test, y_test = train_test(reflectance, label, loc_list)
        x_train_files, y_train_files, x_test_files, y_test_files, loc_test_files, indices, annotation_array_files, tot_sum = save_test_values(x_train, y_train, x_test, y_test, loc_test, annotation_array, x_test_files, y_test_files, loc_test_files, indices, annotation_array_files, x_train_files, y_train_files, tot_sum)
    #print()
    predictions = make_predictions(x_test_files, y_test_files, x_train_files, y_train_files)
    create_classification_maps(predictions, files, annotation_array_files, loc_test_files, indices)