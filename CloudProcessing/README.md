Work completed by Purvi Sehgal in September 2022 under mentorship of Dr. David Thompson

Instructions:

CloudImageProcessing.py is the main file. It trains a model, tests it, creates a classification map, and identifies clouds in a new file.

The model is being tested on only 1/4th of a new file. To change this, change the for loop ranges in create_new_classification_map and get_x_test functions

In if __name__ == "__main__":, there is a variable called number_train_files. Everytime a file is added or removed from EMIT_images.txt, this number must be updated.
It determines how many files will be used to train the model (there will be number_train_files number of classification maps). 
The total number of image names in EMIT_images.txt minus number_train_files is how many files the model is being tested on. For example, if there are 8 image
file names, and number_train_files is 7, then the first 7 are being used to train the model and there will be 7 classification models. 
The model will then be tested on the final image. However, the images being used for testing purposes should have a <imagename>.png file in the directory (rdn).
They should also be vertical (doesn't work on diagonal or pre-transformation images). No annotation file is necessary. 

Annotation files are required for train-test images. These files are transparent except for red (cloud) and green (non_cloud) annotations. 
These files have to be in the format: <imagename>color.png.

Additionally, train-test images and test images both require <imagename>.hdr and <imagename>.img files. The output is annotated<imagename>.png.
For the train-test images the annotated file is the classification map.
For test images the annotated file is the image file with blue being regions predicted as cloud-regions by the program
  
For the classification map, the green pixels are non_cloud train pixels, the red pixels are cloud train pixels.
The yellow pixels are non_cloud test pixels predicted by the model and the blue pixels are cloud test pixels predicted by the model.
  
Irradiance.txt is a file with irradiance values. However, it is no longer being used in the final program. 
Cloudfiles2.csv is another file with file name and solar angle values for the AVIRIS mission. 
It is no longer used but kept in the repository for future uses.

annotated_image.png is a classification map result from a AVIRIS mission cloud identification model with an accuracy of approx 96%.
This was used to test the program before applying it to EMIT.
  
reflectance.png graphs the normalized radiance values as a function of wavelength for the first cloud-annotated pixel for the AVIRIS mission.
  
All files need to be rdn spectral cubes. Moreover, some files are annotated but left out of the program. They can be added in.
There should be a total of 9 annotated files and 2 non-annotated test files. 
One of the 9 annotated files seemed to negatively affect the model's accuracy, so it was removed from EMIT_images.txt.
  
In the ExtraFiles folder, there are multiple types of files. Files starting with ang are from the AVIRIS mission. 
The emit <filename>.png files are non-useable emit files because they contain non-transformed pngs. 
  
In the Pictures folder, ang files are AVIRIS mission useable images. The emit files are EMIT mission useable images. 
These were the images that were annotated for the projects. The ang .txt file has all the wavelengths for the AVIRIS mission for the specific image
This file can be used for graphing.
