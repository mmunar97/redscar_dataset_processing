import os
import cv2
import json
from tqdm import *
from typing import Dict

from staples_detection import StapleDetector
from staples_detection.base.staple_detection_methods import StapleDetectionMethod

from multiprocessing import Pool

def wrapper(params: Dict):
    image_path = params["IMAGE_PATH"]
    gt_path = params["GT_PATH"]
    image_name = params["IMAGE_NAME"]
    image_set = params["SET"]
    saving_path = params["SAVING_PATH"]
    
    image = cv2.imread(image_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(bool)
    
    detector = StapleDetector(image=image)
    discrete_morphology_result = detector.detect_staples(method=StapleDetectionMethod.DISCRETE_MORPHOLOGY,
                                                         ground_truth_mask=gt)
    
    cv2.imwrite(os.path.join(saving_path, image_set, image_name), discrete_morphology_result.final_mask)
    
    try:
        iou = discrete_morphology_result.performance.intersection_over_union()
    except:
        iou = 0
    
    return {"IMAGE_NAME": image_name, 
            "IOU": iou,
            "SET": image_set}
    

if __name__ == "__main__":
    REDSCAR_DATASET_PATH = r"/home/marc/UIB_EXPERIMENTS/REDSCAR/SUBSETS/MACHINE_LEARNING_DATASET"
    SAVING_PATH = r"/home/marc/UIB_EXPERIMENTS/STAPLES_REMOVAL_EXPERIMENTS/DiscreteMorphology"
    
    TRAIN_DATASET = os.path.join(REDSCAR_DATASET_PATH, "train")
    TEST_DATASET = os.path.join(REDSCAR_DATASET_PATH, "test")
    
    with open(os.path.join(REDSCAR_DATASET_PATH, "test.txt")) as test_images_names_file:
        test_images_names = [line.rstrip() for line in test_images_names_file]

    with open(os.path.join(REDSCAR_DATASET_PATH, "train.txt")) as train_images_names_file:
        train_images_names = [line.rstrip() for line in train_images_names_file]
        
    algorithm_params = []    
    for image_name in train_images_names:
        image_path = os.path.join(TRAIN_DATASET, "IMAGES", image_name)
        gt_path = os.path.join(TRAIN_DATASET, "GT_WOUND_MASK", image_name)
        
        algorithm_params.append({"IMAGE_PATH": image_path,
                                 "GT_PATH": gt_path,
                                 "IMAGE_NAME": image_name,
                                 "SET": "train",
                                 "SAVING_PATH": SAVING_PATH})
    for image_name in test_images_names:
            image_path = os.path.join(TEST_DATASET, "IMAGES", image_name)
            gt_path = os.path.join(TEST_DATASET, "GT_WOUND_MASK", image_name)

            algorithm_params.append({"IMAGE_PATH": image_path,
                                     "GT_PATH": gt_path,
                                     "IMAGE_NAME": image_name,
                                     "SET": "test",
                                     "SAVING_PATH": SAVING_PATH})
    
    with Pool() as pool:
        number_results = len(algorithm_params)
        results = [None] * number_results
        
        with tqdm(total=number_results) as progress_bar:
            for x, result in enumerate(pool.imap_unordered(wrapper, algorithm_params)):
                results[x] = result
                progress_bar.update()
    
        with open('iou_results.json', 'w') as fout:
            json.dump(results, fout)