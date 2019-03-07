import cv2
import glob
import json
import numpy as np
import rasterio
from keras.models import Input, Model
from keras.layers import concatenate, Conv2D, Dense, Dropout, GlobalMaxPooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical



def training_mask_generation(img_pan_filename, input_geojson_filename, labels):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_pan_filename: File name or path of panchromatic image to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    - labels: List of labels for multi - class semantic segmentation of image 
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    with rasterio.open(img_pan_filename) as f:
        metadata_pan = f.profile
        img_pan = f.read(1)
    
    mask = np.zeros((img_pan.shape[0], img_pan.shape[1]))
    
    xres = metadata_pan['transform'][0]
    ulx = metadata_pan['transform'][2]
    yres = metadata_pan['transform'][4]
    uly = metadata_pan['transform'][5]
    
    lrx = ulx + (metadata_pan['width'] * xres)                                                         
    lry = uly - (metadata_pan['height'] * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        layer_num = labels.index(str(polygons['features'][polygon]['properties']['Label']))
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((metadata_pan['width']) ** 2 / (metadata_pan['width'] + 1)) / (lrx - ulx)
        yf = ((metadata_pan['height']) ** 2 / (metadata_pan['height'] + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask, position, layer_num)
    
    return np.expand_dims(mask, axis = 2)



def image_clip_to_segment_and_convert(image_ms_array, image_pan_array, mask_array, factor, image_height_size, 
                                      image_width_size):
    """ 
    This function is used to cut up both multispectral and panchromatic images of any input size into segments of a fixed 
    size, with empty clipped areas padded with zeros to ensure that segments are of equal fixed sizes and contain valid data 
    values. The function then returns a 4 - dimensional array containing the entire multispectral and panchromatic images and 
    its mask in the form of fixed size segments. 
    
    Inputs:
    - image_ms_array: Numpy array representing the multispectral image to be used for model training
    - image_pan_array: Numpy array representing the panchromatic image to be used for model training 
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - factor: Ratio of pixel resolution of multispectral image to that of panchromatic image
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    
    Outputs:
    - image_ms_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input multispectral 
                              image array
    - image_pan_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input panchromatic 
                              image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input raster mask
    
    """
        
    img_pan_list = []
    img_ms_list = []
    mask_list = []
    
    n_bands = image_ms_array.shape[2]
    
    for i in range(0, image_pan_array.shape[0] - image_height_size, int(factor)):
        for j in range(0, image_pan_array.shape[1] - image_width_size, int(factor)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            
            img_original = image_pan_array[i : i + image_height_size, j : j + image_width_size, 0]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_pan_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                                 img_flip_both])
            mask_patch = mask_array[i : i + image_height_size, j : j + image_width_size, 0]
            label = mask_patch[int(image_height_size / 2), int(image_width_size / 2)]
            mask_list.extend([label] * 7)
            
    for i in range(0, int(image_ms_array.shape[0] - (image_height_size / factor)), 1):
        for j in range(0, int(image_ms_array.shape[1] - (image_width_size / factor)), 1):
            M_90_ms = cv2.getRotationMatrix2D(((image_width_size / factor) / 2, (image_height_size / factor) / 2), 90, 1.0)
            M_180_ms = cv2.getRotationMatrix2D(((image_width_size / factor) / 2, (image_height_size / factor) / 2), 180, 1.0)
            M_270_ms = cv2.getRotationMatrix2D(((image_width_size / factor) / 2, (image_height_size / factor) / 2), 270, 1.0)
            
            img_original = image_ms_array[i : i + image_height_size, j : j + image_width_size, 0 : n_bands]
            img_rotate_90 = cv2.warpAffine(img_original, M_90_ms, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180_ms, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270_ms, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_ms_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                                 img_flip_both])
                
    image_pan_segment_array = np.zeros((len(img_pan_list), image_height_size, image_width_size, image_pan_array.shape[2]))
    image_ms_segment_array = np.zeros((len(img_ms_list), int(image_height_size / factor), int(image_width_size / factor), 
                                       image_ms_array.shape[2]))
    
    for index in range(len(img_pan_list)):
        image_pan_segment_array[index] = img_pan_list[index]
        image_ms_segment_array[index] = img_ms_list[index]
        
    mask_array = np.array(mask_list)
        
    return image_ms_segment_array, image_pan_segment_array, mask_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, label_list):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - label_list: List containing all the labels to be used for multi - class semantic segmentation (label for background
                  should be in the first position of the list)
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    """
       
    img_ms_files = glob.glob(DATA_DIR + '\\Train_MS' + '\\Train_*.tif')
    img_pan_files = glob.glob(DATA_DIR + '\\Train_Pan' + '\\Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Train_Polygons' + '\\Train_*.geojson')
    
    img_ms_array_list = []
    img_pan_array_list = []
    mask_array_list = []
    
    for file in range(len(img_ms_files)):
        with rasterio.open(img_ms_files[file]) as f:
            metadata = f.profile
            img_ms = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
            
        with rasterio.open(img_pan_files[file]) as g:
            metadata_pan = g.profile
            img_pan = np.expand_dims(g.read(1), axis = 2)
        
        ms_to_pan_ratio = metadata['transform'][0] / metadata_pan['transform'][0]
        
        if (img_height_size % ms_to_pan_ratio) != 0 or (img_width_size % ms_to_pan_ratio) != 0:
            raise ValueError('Please make sure that both img_height_size and img_width_size can be divided by {}'\
                             .format(int(ms_to_pan_ratio)))
        
        mask = training_mask_generation(img_pan_files[file], polygon_files[file], labels = label_list)
        
        img_ms_array, img_pan_array, mask_array = image_clip_to_segment_and_convert(img_ms, img_pan, mask, ms_to_pan_ratio, 
                                                                                    img_height_size, img_width_size)
        
        img_ms_array_list.append(img_ms_array)
        img_pan_array_list.append(img_pan_array)
        mask_array_list.append(mask_array)
        
    img_ms_full_array = np.concatenate(img_ms_array_list, axis = 0)
    img_pan_full_array = np.concatenate(img_pan_array_list, axis = 0)
    mask_full_array = to_categorical(np.concatenate(mask_array_list, axis = 0), num_classes = len(label_list))
    
    return img_ms_full_array, img_pan_full_array, mask_full_array



def TBLCCCNN_Model(pan_image_height_size, pan_image_width_size, ms_to_pan_ratio, n_bands, n1_pan, n2_pan, n3_pan, 
                   n1_ms, n2_ms, n3_ms, dropout_rate, n_classes, l_r):
    """
    This function generates the Two - Branch Land Cover Classification Convolutional Neural Network (TBLCCCNN) that is proposed
    in the paper 'A Two - Branch CNN Architecture for Land Cover Classification of PAN and MS Imagery' by Gaetano R., 
    Ienco D., Ose K., Cresson R. (2018)
    
    Inputs:
    - pan_image_height_size: Height of panchromatic image patches to be used for TBLCCCNN model training
    - pan_image_width_size: Width of panchromatic image patches to be used for TBLCCCNN model training
    - ms_to_pan_ratio: Ratio of pixel resolution of multispectral image to that of panchromatic image
    - n_bands: Number of channels present in the multispectral image
    - n1_pan: Number of filters to be used for the first convolutional layer for the panchromatic branch of the CNN
    - n2_pan: Number of filters to be used for the second convolutional layer for the panchromatic branch of the CNN
    - n3_pan: Number of filters to be used for the third convolutional layer for the panchromatic branch of the CNN
    - n1_ms: Number of filters to be used for the first convolutional layer for the multispectral branch of the CNN
    - n2_ms: Number of filters to be used for the second convolutional layer for the multispectral branch of the CNN
    - n3_ms: Number of filters to be used for the third convolutional layer for the multispectral branch of the CNN
    - dropout_rate: Dropout rate to be used for layers before concatenation operation
    - n_classes: Number of classes to be used for land cover classification (inclusive of background class)
    - l_r: Learning rate to be used for the Adam optimizer
    
    Outputs:
    - 
    
    """
    
    if (pan_image_height_size % ms_to_pan_ratio) != 0 or (pan_image_width_size % ms_to_pan_ratio) != 0:
        raise ValueError('Please make sure that both pan_image_height_size and pan_image_width_size can be divided by \
                          {}'.format(int(ms_to_pan_ratio)))
    
    pan_img_input = Input(shape = (pan_image_height_size, pan_image_width_size, 1))
    conv_1_pan = Conv2D(n1_pan, (7, 7), padding = 'same', activation = 'relu')(pan_img_input)
    max_pool_1_pan = MaxPooling2D(pool_size = (2, 2))(conv_1_pan)
    conv_2_pan = Conv2D(n2_pan, (3, 3), padding = 'same', activation = 'relu')(max_pool_1_pan)
    max_pool_2_pan = MaxPooling2D(pool_size = (2, 2))(conv_2_pan)
    conv_3_pan = Conv2D(n3_pan, (3, 3), padding = 'same', activation = 'relu')(max_pool_2_pan)
    glob_max_pool_pan = GlobalMaxPooling2D()(conv_3_pan)
    glob_max_pool_pan = Dropout(dropout_rate)(glob_max_pool_pan)
    
    ms_img_input = Input(shape = (int(pan_image_height_size / ms_to_pan_ratio), int(pan_image_width_size / ms_to_pan_ratio), 
                                  n_bands))
    conv_1_ms = Conv2D(n1_ms, (3, 3), padding = 'same', activation = 'relu')(ms_img_input)
    conv_2_ms = Conv2D(n2_ms, (3, 3), padding = 'same', activation = 'relu')(conv_1_ms)
    conv_3_ms = Conv2D(n3_ms, (3, 3), padding = 'same', activation = 'relu')(conv_2_ms)
    glob_max_pool_ms = GlobalMaxPooling2D()(conv_3_ms)
    glob_max_pool_ms = Dropout(dropout_rate)(glob_max_pool_ms)
    
    all_features = concatenate([glob_max_pool_pan, glob_max_pool_ms])
    
    pred_layer = Dense(n_classes, activation = 'softmax')(all_features)
    
    tblcccnn_model = Model(inputs = [ms_img_input, pan_img_input], outputs = pred_layer)
    tblcccnn_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = l_r), 
                           metrics = ['categorical_crossentropy'])
    
    return tblcccnn_model



def image_model_predict(input_ms_image_filename, input_pan_image_filename, pan_img_height_size, pan_img_width_size, 
                        fitted_model, write, output_filename):
    """ 
    This function cuts up both panchromatic and multispectral images into segments of fixed size, and feeds each segment to the 
    TBLCCCNN model for land cover classification. The output class is then allocated to its corresponding location in the 
    panchromatic image in order to obtain the complete land cover classification map, after which it can be written to file.
    
    Inputs:
    - input_ms_image_filename: File path of the multispectral image to be pansharpened by the TBLCCCNN model
    - input_pan_image_filename: File path of the panchromatic image to be used by the TBLCCCNN model
    - pan_img_height_size: Height of panchromatic image segment to be used for TBLCCCNN model pansharpening
    - pan_img_width_size: Width of panchromatic image segment to be used for TBLCCCNN model pansharpening
    - ms_to_pan_ratio: The ratio of pixel resolution of multispectral image to that of panchromatic image
    - fitted_model: Keras model containing the trained TBLCCCNN model along with its trained weights
    - write: Boolean indicating whether to write the pansharpened image to file
    - output_filename: File path to write the file
    
    Output:
    - class_layer: Numpy array which represents the land cover classification map for given panchromatic image
    
    """
    
    with rasterio.open(input_ms_image_filename) as f:
        metadata = f.profile
        ms_img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
    
    with rasterio.open(input_pan_image_filename) as g:
        metadata_pan = g.profile
        pan_img = g.read(1)
    
    pan_img = np.expand_dims(pan_img, axis = 2)
    
    ms_to_pan_ratio = metadata['transform'][0] / metadata_pan['transform'][0]
    
    class_layer = np.zeros((pan_img.shape[0], pan_img.shape[1]))
    
    img_pan_holder = []
    img_ms_holder = []
    
    for i in range(0, pan_img.shape[0] - pan_img_height_size, int(ms_to_pan_ratio)):
        for j in range(0, pan_img.shape[1] - pan_img_width_size, int(ms_to_pan_ratio)):
            img_pan_iter = pan_img[i : i + pan_img_height_size, j : j + pan_img_width_size, 0]
            img_pan_holder.append(img_pan_iter)
            
    for i in range(0, int(ms_img.shape[0] - (pan_img_height_size / ms_to_pan_ratio)), int(ms_to_pan_ratio)):
        for j in range(0, int(pan_img.shape[1] - (pan_img_width_size / ms_to_pan_ratio)), int(ms_to_pan_ratio)):
            img_ms_iter = ms_img[i : int(i + (pan_img_height_size / ms_to_pan_ratio)), 
                                 j : int(j + (pan_img_width_size / ms_to_pan_ratio)), 
                                 0 : metadata['count']]
            img_ms_holder.append(img_ms_iter)
            
    img_pan_array = np.concatenate(img_pan_holder, axis = 0)
    img_ms_array = np.concatenate(img_ms_holder, axis = 0)
    
    pred_array = np.argmax(fitted_model.predict([img_ms_array, img_pan_array]), axis = 1)
    
    n = 0 
    for i in range(int(pan_img_height_size / 2), pan_img.shape[0] - int(pan_img_height_size / 2), int(ms_to_pan_ratio)):
            for j in range(int(pan_img_width_size / 2), pan_img.shape[1] - int(pan_img_width_size / 2), int(ms_to_pan_ratio)):
                class_layer[i, j] = pred_array[n]
                n += 1
    
    if write:
        with rasterio.open(output_filename, 'w', **metadata_pan) as dst:
            dst.write(class_layer)
    
    return class_layer