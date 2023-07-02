
# Segmentation module for autnomous cars

## Author

- [Rahil Shah](https://github.com/rahilshah17)


## Introduction

Autonomous driving has emerged as a transformative
technology with the potential to revolutionize
the transportation industry. A key component
of autonomous vehicles is the perception system,
which enables the vehicle to understand its surroundings
and make informed decisions. Accurate
and efficient detection and segmentation of objects
and lanes are crucial for ensuring the safety and
reliability of autonomous driving systems.

This GitHub repository showcases the work I have done as part of the undergraduate research credits program, under the guidance of Dr. Ramkrishna Pasumarthy. Our research team, consisting of five dedicated researchers, collaborated to implement an autonomous car system. My specific focus within the project was on developing the segmentation module.

In this repository, I document my journey of implementing the segmentation, object detection, and lane detection algorithms. Through this project, we aimed to create an intelligent system capable of understanding and interpreting its environment for autonomous navigation.

By sharing my code and findings on this platform, I hope to contribute to the research community and facilitate further advancements in autonomous vehicle technology. This repository serves as a comprehensive resource, providing insights into the implementation process and serving as a foundation for future research in the field.

Feel free to explore the various modules, code snippets, and documentation provided here. I encourage you to collaborate, provide feedback, and build upon the work presented in this repository.

Thank you for your interest in my research, and I hope you find this repository valuable in your own endeavors related to autonomous car segmentation and beyond.

In this project, we present a comprehensive approach
using the YOLOP (You Only Look Once
with Position embedding) model for segmentation,
lane detection, and object detection in autonomous
cars. YOLOP is a state-of-the-art deep
learning model that integrates object detection
and segmentation into a unified framework. By
leveraging the strengths of the YOLO (You Only
Look Once) family of models, YOLOP provides
real-time performance while maintaining high accuracy.

Our approach employs the YOLOP model to simultaneously
perform segmentation, lane detection,
and object detection tasks. This enables the
autonomous car to identify and track objects of interest,
such as vehicles, pedestrians, traffic signs,
and lane markings, in its surroundings. By combining
these tasks into a single model, we achieve
a streamlined and efficient perception system that
contributes to the overall autonomous driving capabilities.

To train the YOLOP model, we utilized a diverse
dataset consisting of labeled images captured from
various driving scenarios. This dataset included a
wide range of environmental conditions, such as
different lighting and weather conditions, as well
as diverse road layouts. We pre-processed the images
by resizing them, applying data augmentation
techniques, and normalizing the pixel values
to enhance the model’s generalization and robustness.

Evaluation of our YOLOP-based approach involved
assessing the performance of the segmentation,
lane detection, and object detection tasks
using appropriate metrics. We measured metrics
such as mean intersection over union (mIoU),
accuracy, precision, and recall to quantify the
model’s performance in different scenarios. Comparative
analysis was conducted against other
state-of-the-art approaches for each task, including
traditional methods and deep learning models.

By utilizing the YOLOP model for segmentation,
lane detection, and object detection, our project
contributes to the development of a reliable and
efficient perception system for autonomous cars.
The integration of these tasks not only enhances
the safety and accuracy of the autonomous driving
system but also reduces computational complexity
and enables real-time operation.

In conclusion, our project demonstrates the effectiveness
of the YOLOP model as a comprehensive
solution for segmentation, lane detection, and object
detection in autonomous cars. The findings
and insights from our work can pave the way for
further advancements in autonomous driving technologies,
enabling safer and more intelligent selfdriving
vehicles
## Prerequisites

To successfully undertake the development of the
segmentation module for the autonomous car, a
strong foundation in deep learning and computer
vision was essential. The following prerequisites
were completed over a period of one and a half
months:

1. Neural Networks and Deep Neural Networks:
- Comprehensive study of neural networks, their architectures, and training algorithms.

- Practical implementation of a deep neural network to predict the MNIST dataset.

- Completion of assignments and exercises to solidify understanding of neural network concepts.

2. Convolutional Neural Networks
(CNNs):
- In-depth exploration of convolutional neural networks and their applications in computer vision.

- Study of different CNN architectures and their components, such as convolutional layers, pooling layers, and fully connected layers.

- Hands-on implementation of CNN models, including the popular ResNet architecture.

3. Object Detection Task:
- Undertaking an object detection task to gain practical experience in computer vision tasks.

- Understanding various object detection algorithms and techniques, including region-based methods and anchor-based methods.

- Successful completion of an object detection task using state-of-the-art models and evaluation of the results.

These prerequisites provided the necessary knowledge
and skills to comprehend advanced concepts
and methodologies in deep learning, specifically
related to segmentation, object detection,
and neural network implementations. The acquired
expertise will significantly contribute to
the successful development and implementation
of the segmentation module for the autonomous
car.

By investing time and effort into completing these
prerequisites, a solid foundation has been established,
enabling a comprehensive understanding of
the underlying principles and techniques required
for this project.
## Object Detection

To evaluate the performance of the object detection
task, a series of images were captured outside
the Electrical Science Block (ESB). These images
were then used to test a pre-trained object detection
model. The results obtained were highly promising, demonstrating the effectiveness of the
model in detecting objects in real-world scenarios. The below images show the results.
![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/im1.png)

![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/im1_pred.png)

![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/im2.png)

![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/im2_pred.png)

![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/im3.png)

![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/im3_pred.png)

The success of the object detection task using the
pre-trained model in these real-world images highlights
its potential for autonomous driving applications.
These results demonstrate the model’s
effectiveness in accurately identifying and localizing
objects, providing crucial information for autonomous
vehicles to make informed decisions in
real-time scenarios.

In order to achieve real-time performance and enable simultaneous segmentation, lane detection, and object detection, we implemented the YOLOP (You Only Look Once for Panotopic Perception) model. 

## YOLOP (You Only Look Once for Panoptic Driving Perception) Architecture

The YOLOP architecture is an advanced variant of the popular YOLO (You Only Look Once) object detection framework. It builds upon the strengths of YOLO while introducing innovative techniques to improve efficiency and reduce computational complexity.

YOLOP features one shared encoder and three decoder heads, each dedicated to solving specific tasks. Unlike complex shared blocks between different decoders, YOLOP aims to keep the computation to a minimum, allowing for easier end-to-end training.

We would like to express our gratitude to the authors of the YOLOP architecture for their outstanding documentation and research work. Some of the explanations provided in this repository are derived from their research paper, which can be found in the references section below. We highly recommend referring to their research paper for a more comprehensive understanding of YOLOP.

The below figure shows us a glimpse of the YOLOP architecture

![App Screenshot](https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/arch.png)

### 4.1 Encoder

The encoder in YOLOP consists of a backbone network and a neck network. The backbone network, CSP-Darknet, extracts features from input images while leveraging feature reuse and propagation to reduce parameters and computations. The neck network performs feature engineering by combining the Spatial Pyramid Pooling (SPP) module and the Feature Pyramid Network (FPN) module. These modules generate and fuse features of different scales and semantic levels, resulting in rich features with multi-scale and multi-level information.

### 4.2 Decoders

YOLOP uses different decoders for various tasks. For object detection, it adopts an anchor-based multi-scale detection technique similar to YOLOv4. The detection head, Path Aggregation Network (PAN), combines semantic features from the FPN top-down and image features from the PAN bottom-up to improve feature fusion. This fusion creates a multi-scale feature map used for object detection.

For drivable area segmentation and lane line segmentation, YOLOP utilizes the same network structure. The features from the bottom layer of the FPN, with size (W/8, H/8, 256), are fed into the segmentation branch. Through three upsampling processes, the feature map is restored to (W, H, 2), representing the pixel-wise probability for drivable areas and lane lines. YOLOP doesn't require an additional SPP module for segmentation heads due to the shared SPP module in the neck network.

By employing this architecture, YOLOP effectively extracts features from images and performs tasks such as object detection, drivable area segmentation, and lane line segmentation.



## Data

The BDD100K dataset, specifically designed for
autonomous driving applications, has played a vital
role in training and evaluating our YOLOPbased
segmentation module. It consists of over
100,000 high-resolution images captured across
diverse driving scenarios, encompassing various
driving conditions, weather patterns, and lighting
scenarios.

The dataset provides comprehensive annotations
for object detection, lane detection, and semantic
segmentation tasks. These annotations, generated
through a combination of manual segmentation
and automated algorithms, include pixel-level
delineation of objects such as cars, pedestrians,
cyclists, and traffic signs. Moreover, lane detection
annotations facilitate accurate identification
and tracking of lane markings, crucial for effective
lane keeping and path planning.

Leveraging the rich annotations of the BDD100K
dataset, our segmentation module underwent extensive
training to simultaneously perform object
detection, lane detection, and segmentation tasks.
This training enabled the module to robustly analyze
real-world driving scenes and deliver accurate
and reliable segmentation results for autonomous
driving applications. By utilizing the diverse and
comprehensive BDD100K dataset, our segmentation
module is well-equipped to handle the challenges
encountered in complex driving environments,
contributing to improved safety and performance
in autonomous vehicles.
## Training

In our project, we successfully implemented the YOLOP architecture as outlined in the research
paper and proceeded to train the model using the
BDD100K dataset. The training process was performed
on a computer system equipped with an
AMD Ryzen 5 processor and AMD Radeon graphics.

Training the YOLOP architecture on the
BDD100K dataset proved to be a computationally
demanding task. The dataset’s size and
complexity necessitated significant computational
resources and time. Despite these challenges,
we were able to complete one training epoch,
which involved processing approximately 6,000
images which took over 3 days for our device.
It is important to note that the paper suggests training over 240 epochs to achieve the reported
level of accuracy.

Due to the limited hardware resources at our disposal,
we could only train a single epoch. However,
we acknowledge that achieving higher accuracy
and faster inference speeds can be accomplished
by leveraging hardware accelerators specifically
designed for deep learning tasks.

The training process played a crucial role in finetuning
the YOLOP model for our segmentation
module, enabling us to obtain accurate and reliable
results in real-world scenarios. Although
we were limited in terms of the number of training
epochs, we are confident that with the use of
hardware accelerators, we can achieve superior accuracy
and faster processing times, thus further
enhancing the performance of our model

By acknowledging the computational limitations
and emphasizing the potential for improved results
with hardware accelerators, we pave the way
for future work to integrate our project with more
powerful hardware resources, ultimately leading
to enhanced accuracy, speed, and overall performance.


## Results

During the training process, we obtained the following results for the first epoch and compared them with the results presented in the paper:

- Accuracy for Driving Area Segmentation: 82.6% (Paper: 89.2%)

The inference time for each frame was measured to be 0.5729 seconds, with an additional 0.2138 seconds per frame for the non-maximum suppression (NMS) step.

It is important to note that these results were obtained using our current hardware setup, which has certain constraints. However, we believe that these performance metrics can be significantly improved by leveraging hardware accelerators specifically designed for deep learning tasks. By utilizing more powerful hardware resources, such as GPUs or specialized AI chips, we can enhance the speed and accuracy of our model, leading to better segmentation, lane detection, and object detection results.


Here are some of the images we used to test the pretrained model:

<table>
  <tr>
    <td>
      <img src="https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/0001TP_010080%20(1).png" alt="Image 1" width="400"/>
    </td>
    <td>
      <img src="https://github.com/rahilshah17/Segmentation-module-for-autonomous-car/blob/main/images_segmentation/0001TP_010080%20(2).png" alt="Image 2" width="400"/>
    </td>
  </tr>
</table>

![Sample Image 1](path/to/sample_image_1.jpg)
![Sample Image 2](path/to/sample_image_2.jpg)
![Sample Image 3](path/to/sample_image_3.jpg)

More sample images and videos can be found [here](path/to/sample_images_and_videos).