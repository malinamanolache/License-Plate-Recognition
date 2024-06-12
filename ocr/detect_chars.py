# solve errors with: "conda install nomkl"
import os
import numpy as np
from datasets import OcrDataset_7chars_plateType
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from collections import Counter
from skimage import io, color, feature, measure, exposure, morphology, filters
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from itertools import combinations
import imageio
#import cv2

def contour_based_segmentation(image):
    smoothed_image = filters.gaussian(image, sigma=2.0)
    
    edges = feature.canny(smoothed_image, sigma=0.5, low_threshold=0.4, high_threshold=1)
    contours = measure.find_contours(edges, level=0.8)
    mask = np.zeros(image.shape, dtype=bool)
    for contour in contours:
        rr, cc = contour[:, 0].astype(int), contour[:, 1].astype(int)
        mask[rr, cc] = 1
    
    filled_mask = binary_fill_holes(mask)
    segmented_image = filled_mask.astype(np.uint8)
    labeled_image, num_objects = measure.label(filled_mask, return_num=True)
    
    return segmented_image, num_objects

def segmnetarea_cu_prag_Otsu(Y, nbins, view_histogram=True):
    # IN: Y este imaginea grayscale si nbins  e nr d ebini de la histograma. Se recomanda un nbins de 100-255.
    # OUT = harta BINARA np.uint8, cu imaginea segmenatta, folosind Otsu
    
    def prag_Otsu(h): # h e de fapt h[0]
        #IN: h e histograma imaginii
        eps=0.000000001
        L=len(h)
        criteriu=np.zeros(L)
        for T in range (0,L):
            P0=0
            mu0=0
            for i in range(0,T):
                P0+=h[i]
                mu0+=i*h[i]
            mu0=mu0/(P0+eps)
    
            P1=0
            mu1=0
            for i in range(T,L):
                P1+=h[i]
                mu1+=i*h[i]
            mu1=mu1/(P1+eps)
        
            criteriu[T]=P0*mu0*mu0+P1*mu1*mu1 # asta e expresia ce se minimizeaza
        
        #plt.figure(),plt.plot(criteriu),plt.show()
        THR = np.argmax(criteriu)
        return THR, criteriu
    
    h=np.histogram(Y,nbins,density=True)
    if view_histogram:
        #print('histogramele:')
        plt.figure(),plt.plot(h[0]),plt.show()
        plt.figure(),plt.plot(h[1][1:], h[0]),plt.show()
        #print('Pragul adaptiv este:')
    prag_adaptiv, _ = prag_Otsu(h[0])
    
    prag_adaptiv_scalat = prag_adaptiv * (np.max(Y)-np.min(Y)) / nbins
    if view_histogram:
        print('Pragul prag_adaptiv_scalat este:')
        print(prag_adaptiv_scalat)
    
    
    img_seg = np.uint8(Y<prag_adaptiv_scalat)
    
    return img_seg

def pastreaza_doar_obiectele_mai_mari_de_procent(BW, p=2):
    # IN: BW este BINARA, cu 0 valoarea fundalului.
    
    L, C = BW.shape
    [LabelImage, nums]=measure.label(BW,return_num='True')
    # vcalculez care e nr minim de pixelo  = 2% din pixelii imaginii, ca sa stiu ce obiecte admit
    nr_pixeli_minim = (p/100)*(L*C)
    #print('Nr pixeli minim')
    #print(nr_pixeli_minim)

    # creez o lista cu indicii obiectelor care depasesc 2%
    lista_obiecte_admise = []
    
    '''
    # COLOCVIU! ATENTIE! Etichetarea obiectelor care nu sunt fundal porneste de la 1
    for i in range(1, nums+1):
        # ! Se va merge pe PRESUPUNEREA ca o sa am 1 pe pozitia obiectulu si 0 in rest, deci, pt a afla NR PIXELI, fac SUMA
        if np.sum(np.uint8(LabelImage == i)) > nr_pixeli_minim:
            lista_obiecte_admise.append(i)
    #print('Lista de obiecte admise:')
    #print(lista_obiecte_admise) # concluzie: s-a admis un singur obiect
    '''
    
    ALLPROPS=measure.regionprops(LabelImage)
    i = 0
    for prop in ALLPROPS:
        if prop.area > nr_pixeli_minim:
            lista_obiecte_admise.append(i+1)
        i += 1
    

    # acum extrag obiectul:
    COMP = np.uint8(np.zeros((L, C)))
    for i in lista_obiecte_admise:
        COMP = COMP + np.uint8(LabelImage == i)
    
    #plt.figure(), plt.imshow(COMP, cmap='gray'), plt.show()
    return COMP, len(lista_obiecte_admise)


# def find_objects_with_aspect_ratio(BW, inf_aspect_ratio, sup_aspect_ratio):
#     L, C = BW.shape
#     [LabelImage, nums]=measure.label(BW,return_num='True')
#     all_props = measure.regionprops(LabelImage)
#     indexes = []
    
#     for i, prop in enumerate(all_props):
#         aspect_ratio = prop.major_axis_length / (prop.minor_axis_length + 0.0001)
#         #print(aspect_ratio)
#         if inf_aspect_ratio <= aspect_ratio and aspect_ratio <= sup_aspect_ratio:
#             indexes.append(i + 1)  # Note: regionprops indexing starts at 1

#     COMP = np.uint8(np.zeros((L, C)))
#     for i in indexes:
#         COMP = COMP + np.uint8(LabelImage == i)
            
#     return COMP, indexes

def find_objects_with_aspect_ratio(BW, inf_aspect_ratio, sup_aspect_ratio):
    L, C = BW.shape
    LabelImage, nums = measure.label(BW, return_num=True)
    all_props = measure.regionprops(LabelImage)
    indexes = []

    for i, prop in enumerate(all_props):
        # Calculate the aspect ratio based on bounding box
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = height / (width + 0.0001)
        
        # Check if the object meets the aspect ratio criteria and is taller than it is wide
        if inf_aspect_ratio <= aspect_ratio <= sup_aspect_ratio and height > width:
            indexes.append(i + 1)  # Note: regionprops indexing starts at 1

    COMP = np.uint8(np.zeros((L, C)))
    for i in indexes:
        COMP = COMP + np.uint8(LabelImage == i)
            
    return COMP, indexes

# def filter_by_alignment(BW):
#     L, C = BW.shape
#     LabelImage, nums = measure.label(BW, return_num=True)
#     all_props = measure.regionprops(LabelImage)
#     indexes = []

#     # Filter objects by aspect ratio
#     for i, prop in enumerate(all_props):
#         indexes.append((i + 1, prop.centroid[1]))  # Store index and centroid x-coordinate

#     # Sort objects by their x-coordinate (left to right)
#     indexes_sorted = sorted(indexes, key=lambda x: x[1])

#     # Select the top 7 most aligned objects
#     top_7_indexes = [idx for idx, _ in indexes_sorted[:7]]

#     # Create a binary image with the selected objects
#     COMP = np.uint8(np.zeros((L, C)))
#     for i in top_7_indexes:
#         COMP = COMP + np.uint8(LabelImage == i)
    
#     return COMP, top_7_indexes

def filter_by_alignment(BW):
    '''
    def find_top_similar_pairs(SIMILARITY_MATRIX, top_n=7):
        # Set the diagonal elements to a very large number
        np.fill_diagonal(SIMILARITY_MATRIX, np.inf)

        # Get the indices of the matrix elements sorted by value (in ascending order)
        sorted_indices = np.unravel_index(np.argsort(SIMILARITY_MATRIX, axis=None), SIMILARITY_MATRIX.shape)
        print('sorted_indices: ')
        print(sorted_indices)
        # Extract the top N pairs
        top_pairs = list(zip(sorted_indices[0], sorted_indices[1]))[:top_n]
        print('top_pairs: ')
        print(top_pairs)
        
        return top_pairs
    '''

    def find_best_group_of_7(SIMILARITY_MATRIX):
        # Number of items
        n = SIMILARITY_MATRIX.shape[0]
        
        # Generate all possible groups of 7 indices
        all_combinations = list(combinations(range(n), 7))
        
        best_group = None
        best_similarity_sum = np.inf  # Initialize to a large number for minimization

        # Evaluate each group
        for group in all_combinations:
            similarity_sum = 0
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    similarity_sum += SIMILARITY_MATRIX[group[i], group[j]]

            if similarity_sum < best_similarity_sum:
                best_similarity_sum = similarity_sum
                best_group = group


        return best_group

    L, C = BW.shape
    LabelImage, nums = measure.label(BW, return_num=True)
    all_props = measure.regionprops(LabelImage)
    indexes = []

    # Filter objects by aspect ratio
    for i, prop in enumerate(all_props):
        # Calculate the aspect ratio based on bounding box
        minr, minc, maxr, maxc = prop.bbox
        
        #indexes.append((i + 1, minc, maxc))
        indexes.append((i + 1, minr, maxr))

    # Sort objects by their leftmost x-coordinate (bounding box limit)
    indexes_sorted = sorted(indexes, key=lambda x: x[1])

    # Select the top 7 most aligned objects
    #top_7_indexes = [idx for idx, _ in indexes_sorted[:7]]
    SIMILARITY_MATRIX = np.zeros((len(indexes), len(indexes)))
    for i in range(len(indexes)):
        for j in range(len(indexes)):
            # **** mai jos: suma diferentelor dintre cooordonatele bbox de oe Oy si de pe Ox
            # **** un scor mai mic inseamna similaritate mai mare
            SIMILARITY_MATRIX[i, j] = abs(indexes[i][1] - indexes[j][1]) + abs(indexes[i][2] - indexes[j][2])

    #top_similar_pairs = find_top_similar_pairs(SIMILARITY_MATRIX)
    selected_indexes = find_best_group_of_7(SIMILARITY_MATRIX)
    #print(selected_indexes)
    #print('selected_indexes: ', selected_indexes)

    selected_indexes = {index+1 for index in selected_indexes}
    #print(top_similar_pairs)

    # *********************************
    # GRESEALA: nu vreau cele mai similare 7 perechi vreau CELE MAI SIMILARE 7 OBIECTEEEE
    # *********************************

    # Create a binary image with the selected objects
    COMP = np.uint8(np.zeros((L, C)))
    for i in selected_indexes:
        COMP = COMP + np.uint8(LabelImage == i)
    
    return COMP, selected_indexes

def display(image):
    #print(np.transpose(image.numpy(), (1, 2, 0)).shape)
    image_pil = Image.fromarray((np.transpose(image.numpy(), (1, 2, 0)) * 255).astype(np.uint8)) 
    image_pil.show()

def try_again(Y, clip_limit):
    Y_clahe = exposure.equalize_adapthist(Y / 255.0, clip_limit=clip_limit)
    Y_clahe = (255 * Y_clahe).astype(np.uint8)
    Y_thresh = segmnetarea_cu_prag_Otsu(Y_clahe, 100, view_histogram=False)

    COMP, nr_obiecte = pastreaza_doar_obiectele_mai_mari_de_procent(Y_thresh, 0.5)

    return COMP, nr_obiecte


def try_again_with_contour(Y, clip_limit):
    Y_clahe = exposure.equalize_adapthist(Y / 255.0, clip_limit=clip_limit)
    Y_clahe = (255 * Y_clahe).astype(np.uint8)

    COMP, nr_obiecte = contour_based_segmentation(Y_clahe)

    return COMP, nr_obiecte

'''def extract_sorted_objects(binary_image):
    labeled_image, num_objects = measure.label(binary_image, return_num=True)
    regions = measure.regionprops(labeled_image)
    
    objects_with_bboxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cropped_object = binary_image[minr:maxr, minc:maxc]
        objects_with_bboxes.append((minc, cropped_object))
    
    objects_with_bboxes.sort(key=lambda x: x[0])
    sorted_objects = [obj for _, obj in objects_with_bboxes]
    
    return sorted_objects, num_objects'''

def extract_sorted_objects(binary_image, grayscale_image):
    labeled_image, num_objects = measure.label(binary_image, return_num=True)
    regions = measure.regionprops(labeled_image)
    
    objects_with_bboxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        
        # Calculate expanded bounding box
        height = maxr - minr
        expanded_minr = max(0, minr - int(0.4 * height))  # 40% upwards
        expanded_maxr = min(grayscale_image.shape[0], maxr + height)  # 100% downwards
        
        # Crop the object from the grayscale image using the expanded bounding box
        cropped_object = grayscale_image[expanded_minr:expanded_maxr, minc:maxc]
        objects_with_bboxes.append((minc, cropped_object))
    
    # Sort the objects based on the x-coordinate
    objects_with_bboxes.sort(key=lambda x: x[0])
    sorted_objects = [obj for _, obj in objects_with_bboxes]
    
    return sorted_objects, num_objects

def extract_detection_boxes(image, verbose=False):
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image = np.transpose(image.numpy(), (1, 2, 0))
    #print(image.shape)
    Y = np.uint8(255 * color.rgb2gray(image))
    if verbose:
        plt.figure(figsize=(10, 10)), plt.imshow(Y, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()
    Y_clahe = exposure.equalize_adapthist(Y / 255.0, clip_limit=0.015)
    Y_clahe = (255 * Y_clahe).astype(np.uint8)
    if verbose:
        plt.figure(figsize=(10, 10)), plt.imshow(Y_clahe, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()
    Y_thresh = segmnetarea_cu_prag_Otsu(Y_clahe, 100, view_histogram=False)

    COMP, nr_obiecte = pastreaza_doar_obiectele_mai_mari_de_procent(Y_thresh, 0.5)
    if verbose:
        plt.figure(figsize=(10, 10)), plt.imshow(COMP, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()

    if nr_obiecte < 7:
        # first of all try to invert the image - maybe this will work:
        Y_inverted = 255 - Y
        clip_limit = 0.015
        COMP, nr_obiecte = try_again(Y_inverted, clip_limit)
        if verbose:
            plt.figure(figsize=(10, 10)), plt.imshow(COMP, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
        if nr_obiecte < 7:
            clip_limit = 0.06
            COMP, nr_obiecte = try_again(Y, clip_limit)
            if verbose:
                print("with 0.06")
                plt.figure(figsize=(10, 10)), plt.imshow(COMP, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
            if nr_obiecte < 7:
                clip_limit = 0.03
                COMP, nr_obiecte = try_again_with_contour(Y, clip_limit)
                #print(nr_obiecte)
                if nr_obiecte < 7:
                    if verbose:
                        print('Found less than 7 objects')
                    return None, None, None
    COMP2, idxs2 = find_objects_with_aspect_ratio(COMP, 1, 25)
    if verbose:
        plt.figure(figsize=(10, 10)), plt.imshow(COMP2, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
    if len(idxs2) < 7:
        if verbose:
            print('Found less than 7 objects')
        return None, None, None
    COMP3, idxs3 = filter_by_alignment(COMP2)
    print(idxs3)
    if verbose:
        plt.figure(figsize=(10, 10)), plt.imshow(COMP3, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
    if len(idxs3) < 7:
        if verbose:
            print('Found less than 7 objects')
        return None, None, None
    objects, number_objects = extract_sorted_objects(COMP3, Y_clahe)
    return COMP3, objects, number_objects

if __name__ == "__main__":
    dataset_dir = r'/home/radu/fac/ai2/proiect_ia/Our_project (2)(1)/datasets/rodosol'
    dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='training', augmentation_zaga=False, augmentation_overlap=False, augmentation_rotate=False)
    '''
    image, plate_number_tensor, plate_number, plate_type = dataset[2004]
    #image, plate_number_tensor, plate_number, plate_type = dataset[3000]
    #image, plate_number_tensor, plate_number, plate_type = dataset[500]
    #image, plate_number_tensor, plate_number, plate_type = dataset[200] # asta are un 1 si e foarte stearsa
    #image = np.transpose(image.numpy(), (1, 2, 0))
    extract_detection_boxes(image, verbose=True)
    '''
    '''
    print(image.shape)
    Y = np.uint8(255 * color.rgb2gray(image))
    #plt.figure(figsize=(10, 10)), plt.imshow(image, vmin=0, vmax=255), plt.colorbar(), plt.show()
    plt.figure(figsize=(10, 10)), plt.imshow(Y, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()
    Y_thresh = segmnetarea_cu_prag_Otsu(Y, 100, view_histogram=False)
    plt.figure(figsize=(10, 10)), plt.imshow(Y_thresh, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()

    COMP, nr_obiecte = pastreaza_doar_obiectele_mai_mari_de_procent(Y_thresh, 0.5)
    plt.figure(figsize=(10, 10)), plt.imshow(COMP, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
    COMP2, idxs2 = find_objects_with_aspect_ratio(COMP, 1, 25)
    print(idxs2)
    plt.figure(figsize=(10, 10)), plt.imshow(COMP2, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
    COMP3, idxs3 = filter_by_alignment(COMP2)
    print(idxs3)
    plt.figure(figsize=(10, 10)), plt.imshow(COMP3, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
    '''

    dataset_dir = r'/home/radu/fac/ai2/proiect_ia/Our_project (2)(1)/datasets/rodosol'
    dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='training', augmentation_zaga=False, augmentation_overlap=False, augmentation_rotate=False)
    '''
    for idx in [1245, 1246, 1248, 1252, 1259, 1266, 1267, 1268, 1275, 1278, 1284, 1285, 1287, 1295, 1298, 1301, 1302, 1303, 1304, 1306, 1309, 1315, 1318, 1320, 1321, 1322, 1326, 1329, 1331, 1332, 1333, 1337, 1338, 1340, 1341, 1352, 1356, 1358, 1360, 1362, 1366, 1367, 1368, 1372, 1373, 1377, 1380, 1381, 1383, 1384, 1385, 1387, 1391, 1392, 1395, 1396, 1398, 1400, 1401, 1409,]:
        image, plate_number_tensor, plate_number, plate_type = dataset[idx]
        #image_cv2 = (cv2.cvtColor(np.transpose(image.numpy(), (1, 2, 0))* 255, cv2.COLOR_RGB2BGR)).astype(np.uint8)
        segmented_image = extract_detection_boxes(image, verbose=True)
        if segmented_image is not None:
            plt.figure(figsize=(10, 10)), plt.imshow(segmented_image, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()
    '''
    seven_object_images = 0
    NOT_object_images = 0
    failed_indexes = []
    
    output_dir = '/home/radu/fac/ai2/proiect_ia/Our_project (2)(1)/datasets/rodosol_char_boxes/training'
    
    for idx in tqdm(range(len(dataset))):
        image, plate_number_tensor, plate_number, plate_type = dataset[idx]
        #image_cv2 = (cv2.cvtColor(np.transpose(image.numpy(), (1, 2, 0))* 255, cv2.COLOR_RGB2BGR)).astype(np.uint8)
        segmented_image, objects, number_objects = extract_detection_boxes(image, verbose=False)
        if segmented_image is not None:
            seven_object_images += 1
            for i in range(number_objects):
                #print(plate_number)
                correct_label = plate_number[i]
                save_path = os.path.join(output_dir, 'training_idx_' + str(idx) + '_' + str(i) + '_class_' + correct_label + '.png')
                imageio.imwrite(save_path, objects[i] * 255)
        else:
            NOT_object_images += 1
            failed_indexes.append(idx)

    print(failed_indexes)
    print('Total images: ', len(dataset))
    print('seven_object_images: ', seven_object_images)
    print('NOT_object_images: ', NOT_object_images)
    

    plt.hist(failed_indexes, bins=1000, edgecolor='black')

    image_list = [2000, 100, 40, 3033, 209, 1900]
    for i in range(len(dataset)):
        if i % 9 == 0:
            image_list.append(i)


    for idx in image_list:
        image, plate_number_tensor, plate_number, plate_type = dataset[idx]
        #image_cv2 = (cv2.cvtColor(np.transpose(image.numpy(), (1, 2, 0))* 255, cv2.COLOR_RGB2BGR)).astype(np.uint8)
        segmented_image, objects, number_objects = extract_detection_boxes(image, verbose=True)
        if segmented_image is not None:
            plt.figure(figsize=(10, 10)), plt.imshow(segmented_image, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()