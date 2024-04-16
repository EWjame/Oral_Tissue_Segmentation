# Oral Tissues Segmentation
### Project Name : Comprehensive Segmentation Algorithms for Oral Images 
### Paper Link : (Not published yet)
This repository Oral Tissues Segmentation is my master's degree Thesis that using Segmentation models to Segment oral tissue in oral images into 8 parts include :(Oral mucosa, Lips, Tongue, teeth, Floor of Mouth, Gingiva, Others Inside and Background)


Ground Truth Example 
![alt text](https://github.com/EWjame/Oral_Tissue_Segmentation/blob/main/images/ground%20truth.png)

Experiment Objective: </br>
(1) Developing deep learning models for segmenting oral tissues into 8 elements (tissues). </br>
(2) Locating the oral tissue in oral images to narrow the area for consideration. </br>
(3) Testing the hypothesis if the oral cropping algorithm has an effect on the efficiency of oral tissue segmentation. </br>
(4) Testing the hypothesis if the image resizing may affect theoral tissue segmentation performance. </br>

Each objective corresponds to one experiment, so this research will comprise a total of 4 experiments.

We use 4 different state-of-the-arts Segmentation models and 3 different Backbones for training and testing in 1st Experiment. </br>
Models : U-Net, Feature Pyramid Networks (FPN), LinkNet and DeeplapV3+ </br>
Backbones(No.params) : EfficientNetB5(30.6M), ResNet50(25.6M) and MobileNetV2(3.5M)
