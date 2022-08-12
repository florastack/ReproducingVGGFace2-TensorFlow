# ReproducingVGGFace2/TensorFlow
In machine learning, facial recognition aims to identify or verify a person based on an image, often necessitating different subtasks, including facial detection, where the face is found within an image. This paper and reproduction focus on identification as a classification task among identities and facial verification as a regression task that produces a vector (“facial embedding”) for a given face. This embedding may then be used to discern if a photo is of the same person, based on a distance calculation from other facial embeddings. 

Cao Q et al (2018) shows how images from the VGGFace2 dataset were collected, curated, and prepared prior to modelling to train and evaluate face recognition models. The models have been trained on the dataset, namely a ResNet-50 model and a SqueezeNet-ResNet-50 model (referred to as SE-ResNet-50 or SENet) and have been evaluated on standard face recognition data sets, then demonstrating state-of-the-art performance. Also, two claims are defined in the studied paper concerning age and racial variation, which will be further discussed in the following sections. 
This work attempts to contribute an image dataset that contributes a range of images over pose, age and ethnicity. We will explore this dataset’s claims: 
Claim 1: variation in age representation can improve performance 
Claim 2: the dataset is designed to focus more on the diversities between classes, which impairs the matching performance between different poses and ages (also discussed but not of main importance as Claim 1) 
We utilized Rekin Can Malli Keras implementation over TensorFlow, available on Github. This included the model’s pre-trained weights. Not having direct access to which weights the model was trained on posed its own problems, as detailed in the experiments. All experiments were run in Google Colab, utilizing 1 GPU. 

In experiment 1, the test set of vggface2 was acquired, hosted, and downloaded to the colab instance’s local filesystem. Experiment 1 sought to replicate the paper’s template comparison strategy: The authors annotated a subset of the vggface2 test set containing 100 subjects, with 20 images each, split into 4 templates of 2 labels, ‘young’ and ‘mature’. We created a version of the 2x2 similarity matrices for each subject, where each element is the cosine similarity between two templates - enabling us to compare the effect of age. We note that the templates are defined by the position in the age subset image’s list, which is not easily discerned from the metadata, and that a single subject was omitted due to missing image files. Due to missing files in the vggface2 test set, we omitted 1 of 100 subjects during template construction. We could not use the RCMalli implementation for the original VggFace comparison and instead relied on the M Lewis TensorFlow implementation. 

In experiment 2, Multi-Task Cascaded Convolutional Neural Network, or MTCNN, is used for face finding and extracting faces from photos. Assuming only one face in the photo, square output face with the shape 224×224 is extracted from a given photo and is then used as input for face identification. Our Keras model is used directly to predict the probability of a given face belonging to one or more known celebrities. Then the class integers can be mapped to the names of the celebrities. The five names with the highest probability are retrieved. 

References:

Cao Q, Shen L, Xie W, Parkhi O and Zisserman A (2018). Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018), pages 67–74. IEEE, 201 

Can R (2020) VGGFace implementation with Keras Framework. Obtained from GitHub: 
https://github.com/rcmalli/keras-vggface 

Brownlee J (2020) How to Perform Face Recognition With VGGFace2 in Keras. Obtained from post: 
https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-netw ork-in-keras/

Simonyan K and Zisserman A (2015) Very Deep Convolutional Networks for Large-Scale Image Recognition. Obtained from: https://arxiv.org/abs/1409.1556 
Teplyuk A (2019) Author states to have copied from Nguyen K. VGGFace baseline in Keras. Obtained from: https://www.kaggle.com/code/ateplyuk/vggface-baseline-in-keras/notebook 

Zhang K, Zhang Z, Li Z and Qiao Y (2016) Joint face detection and alignment using multitask cascaded convolutional network. Obtained from: https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf 

Yucer S, Akcay S, Al-Moubayed N and Breckon T (2020) Exploring Racial Bias within Face Recognition via per-subject Adversarially-Enabled Data Augmentation. Obtained from: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Yucer_Exploring_Racial_Bias_Within_Face_ Recognition_via_Per-Subject_Adversarially-Enabled_Data_CVPRW_2020_paper.pdf 

Rajesh P (2020) Building Face Recognition Model Under 30 Minutes. Hands-on tutorial: 
https://towardsdatascience.com/building-face-recognition-model-under-30-minutes-2d1b0ef72fda 

Gwilliam M, Hegde S, Tinubu L and Hanson A (2021) Rethinking Common Assumptions to Mitigate Racial Bias in Face Recognition Datasets. Obtained from: https://arxiv.org/pdf/2109.03229.pdf 

Ngo QT, Yoon S. Facial Expression Recognition Based on Weighted-Cluster Loss and Deep Transfer Learning Using a Highly Imbalanced Dataset. Sensors. 2020; 20(9):2639. Obtained from: https://doi.org/10.3390/s20092639 
