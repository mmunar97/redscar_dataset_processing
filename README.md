# Redscar Dataset Processing

This repository is dedicated to store all the Jupyter notebooks used to generate the baseline methods and their results introduced in the article describing the Redscar Dataset (accessible from the [official Redscar website](http://redscar.uib.es/)), which is:

```
Munar M., González-Hidalgo M, Moyà-Alcover G. 
Developing a Dataset for automatic follow-up of abdominal post-surgical wounds: Redscar
``` 

In that paper we propose some baseline methods to solve the following tasks: wound segmentation, staples segmentation and infection classification. The used notebooks and pre-trained models are described in the following sections.

## Notebooks and models for wound segmentation

To perform the segmentation of the wound present in the images of the Redscar Dataset, three different neural network architectures have been trained (available in [this repository](https://github.com/mmunar97/cnn_architectures)), which in turn have been trained with different parameters (number of epochs, input dimension, etc.). Specifically, the experiments are: 

* Segmentation using the U-Net architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_WoundSegmentation_UNet.ipynb`. All the models have 65 filters, an input size of $512\times 512\times 3$ and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - UNet model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT1-WOUND/unet1_epochs%3D25_lr%3D3e-4_res%3D0.h5/download).
  - UNet model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT1-WOUND/unet2_epochs%3D55_lr%3D3e-4_res%3D0.h5/download).
  - UNet model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT1-WOUND/unet3_epochs%3D100_lr%3D3e-4_res%3D0.h5/download).

* Segmentation using the Double U-Net architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_WoundSegmentation_DoubleUNet.ipynb`. All the models have an input size of $256\times 256\times 3$ and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - Double U-Net model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT1-WOUND/dun1_epochs%3D25_lr%3D3e-5_res%3D0.h5/download).
  - Double U-Net model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT1-WOUND/dun2_epochs%3D55_lr%3D3e-5_res%3D0.h5/download).
  - Double U-Net model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT1-WOUND/dun3_epochs%3D100_lr%3D3e-5_res%3D0.h5/download).

* Segmentation using the GSC architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_WoundSegmentation_GSC.ipynb`. All the models have an input size of $512\times 512$ (grayscale) and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - GSC model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT1-WOUND/gsc1_epochs%3D25_lr%3D3e-5_res%3D0.h5/download).
  - GSC model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT1-WOUND/gsc2_epochs%3D55_lr%3D3e-5_res%3D0.h5/download).
  - GSC model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT1-WOUND/gsc3_epochs%3D100_lr%3D3e-5_res%3D0.h5/download).

## Notebooks and models for staples segmentation

To perform the segmentation of the staples present in the images of the Redscar Dataset, three different neural network architectures have been trained (available in [this repository](https://github.com/mmunar97/cnn_architectures)), which in turn have been trained with different parameters (number of epochs, input dimension, etc.). Specifically, the experiments are: 

* Segmentation using the U-Net architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_StaplesSegmentation_UNet.ipynb`. All the models have 65 filters, an input size of $512\times 512\times 3$ and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - UNet model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT2-STAPLES/unet1_epochs%3D25_lr%3D3e-4_res%3D0.h5/download).
  - UNet model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT2-STAPLES/unet2_epochs%3D55_lr%3D3e-4_res%3D0.h5/download).
  - UNet model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT2-STAPLES/unet3_epochs%3D100_lr%3D3e-4_res%3D0.h5/download).

* Segmentation using the Double U-Net architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_StaplesSegmentation_DoubleUNet.ipynb`. All the models have an input size of $256\times 256\times 3$ and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - Double U-Net model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT2-STAPLES/dun1_epochs%3D25_lr%3D3e-5_res%3D0.h5/download).
  - Double U-Net model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT2-STAPLES/dun2_epochs%3D55_lr%3D3e-5_res%3D0.h5/download).
  - Double U-Net model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT2-STAPLES/dun3_epochs%3D100_lr%3D3e-5_res%3D0.h5/download).

* Segmentation using the GSC architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_StaplesSegmentation_GSC.ipynb`. All the models have an input size of $512\times 512$ (grayscale) and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - GSC model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT2-STAPLES/gsc1_epochs%3D25_lr%3D3e-5_res%3D0.h5/download).
  - GSC model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT2-STAPLES/gsc2_epochs%3D55_lr%3D3e-5_res%3D0.h5/download).
  - GSC model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT2-STAPLES/gsc3_epochs%3D100_lr%3D3e-5_res%3D0.h5/download).

## Notebooks for infection classification

As indicated in the paper, the steps performed in the baseline method to detect infection are as follows:

1. Determination of the wound region. This is carried out using the Double U-Net trained with 100 epochs for wound segmentation.
2. Detection and removal of staples from the scene. For the segmentation of staples the U-Net model trained with 100 epochs is applied, while for the removal of staples from the scene the Navier-Stokes' inpainting method is applied. 
  - The package gathering several inpainting methods is [inPYinting](https://github.com/mmunar97/inPYinting). 
4. Chromatic staple-free wound segmentation. Chromatic segmentation of the image using fuzzy sets is applied, considering different colour palettes of the HSV colour space.
  - The package gathering several colour segmentation methods is [ColourSegmentation](https://github.com/mmunar97/colour-segmentation). 
6. Achromatic colour detection. After chromatic segmentation of the image with step 3), those colours that are achromatic (black, grey and white) are detected and oversampled over the previous segmentation. 
7. Redness ratio calculation. It calculates how many pixels have been classified as red with respect to the total number of pixels in the image. 

Having said that, the notebook that calculates all the redness ratios using this process is located at `notebooks/infection_detection/InfectionDetection.ipynb`. Then, the redness ratio obtained for each palette can be found in the following files:
* For the train set, is located at `notebooks/infection_detection/train_redness_evaluation.json`.
* For the test set, is located at `notebooks/infection_detection/test_redness_evaluation.json`.

Then, in order to train the different models to classify whether or not an image has infection based on the redness ratios of the different palettes considered, we have the file written with the `R` language that reads and processes this information, which is located at `notebooks/infection_detection/Analysis.R`.
