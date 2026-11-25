# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 08:31:47 2023

@author: stemmler_t

Modified on Wed Nov 18 2026
@modified by: Samer Alashhab (salashhab@ua.es)
"""


"""
--------------------------------------
IMPORT LIBS
--------------------------------------

"""

import logging
logger = logging.getLogger("root.ClassifLogger")
logger.debug("ClassifLogger has been initialized")

import cv2
import numpy as np
from tensorflow.keras import models
import os

import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn.functional as F

import random

"""
--------------------------------------
Classes
--------------------------------------

"""

class PomoClassification:

    def __init__(self, pathModel, pollenTreshold):
        self.pathModel = pathModel
        
        self.speciesNames = pollenTreshold

        # Disable gradient computation
        torch.set_grad_enabled(False)

        # Set deterministic behavior
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if os.path.isfile(pathModel):
            # Get version
            head, tail = os.path.split(self.pathModel)
            version = "V3.2"
            if version.startswith("v"):
                self._modelVersion = version
            else:
                logger.warning("Could not get version of classifier")
                self._modelVersion = "vX.X"
            logger.info(f"Load classifier model: {tail}")

            self._IMAGE_SIZE = 350

            self._transform = transforms.Compose([
            transforms.Resize((self._IMAGE_SIZE, self._IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
           ])

            # Load model
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print(f"Number of classes: {len(self.speciesNames)}")

            # Create the model with the correct number of classes
            self._model = timm.create_model("convnext_base", pretrained=False, num_classes=len(self.speciesNames))

            # Load checkpoint
            checkpoint = torch.load(self.pathModel, map_location=self._device)

            # Load model weights
            load_result = self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Move to device
            self._model = self._model.to(self._device)
            self._model.eval()

        else:
            logger.error("File/path of classifier model is not existing")
            raise ValueError("File/path of classifier model is not existing")


    def setSpeciesNames(self, Spec:list):
        self.speciesNames = Spec
       
 

    def classifyObj(self, cuttedObj,imagefolder, print_klassifikation=True):
            
        imgTensors = []

        originalImages = []

        for i, obj in enumerate(cuttedObj):
            # Convert grayscale to RGB if needed
            if len(obj.shape) == 2 or obj.shape[2] == 1:
                obj = cv2.cvtColor(obj, cv2.COLOR_GRAY2RGB)

            # Convert BGR (OpenCV default) to RGB
            obj_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)

            # Convert to PIL image
            img_pil = Image.fromarray(obj_rgb)

            # Apply transform
            tensor = self._transform(img_pil)
            imgTensors.append(tensor)
            originalImages.append(obj.copy())
            

        if not imgTensors:
            return []  # no input case


        # Stack all tensors into a batch
        batch = torch.stack(imgTensors).to(self._device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(batch)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, dim=1)

        classifiedObjs = []


        for i, probs in enumerate(probabilities):
            # Top-1 prediction
            indexFirst = torch.argmax(probs).item()
            classifScore = torch.round(probs[indexFirst] * 10000) / 10000

            # Top-2 prediction
            probs_clone = probs.clone()
            probs_clone[indexFirst] = float('-inf')
            indexSecond = torch.argmax(probs_clone).item()
            classifScoreSecond = torch.round(probs[indexSecond] * 10000) / 10000

            # Get class labels
            labelFirst = self.speciesNames[indexFirst]
            labelSecond = self.speciesNames[indexSecond]


            # Store classification info
            classifiedObjs.append({
                'species': labelFirst,
                'index': indexFirst,
                'hitRate': classifScore,
                'speciesSecond': labelSecond,
                'indexSecond': indexSecond,
                'hitRateSecond': classifScoreSecond
            })
            
            print(f"[Image {i}] Class: {indexFirst}, Confidence: {classifScore:.4f} | Second: {indexSecond}, {classifScoreSecond:.4f}")

        return classifiedObjs


if __name__ == '__main__':
    
    classif = PomoClassification(r"")
    classif.loadClassNames(r"")
    
    path = r""
    for img in os.listdir(path):
        if os.path.isfile(os.path.join(path,img)):
            img = cv2.imread(os.path.join(path,img))
            res = classif.classifyObj([img])
            res = res[0]
            
            cv2.imwrite(os.path.join(path,"out", res["species"]["ger"] + "_" + str(res["hitRate"]) + ".png"), img)
