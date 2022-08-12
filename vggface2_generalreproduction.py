import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

!pip install 'h5py<3.0.0' #important to use old version because there's a bug in keras

"""Tensorflow"""

# import tensorflow
# print(tensorflow.__version__)

!pip install tensorflow==2.2.0

#from tensorflow.python.keras import datasets

"""Keras: 

Download framework and VggFace2 implementation. May have to run the installation cell of keras twice to get correct version.
"""

# !pip3 uninstall keras-nightly
# !pip3 uninstall -y tensorflow
!pip3 install keras==2.2.5
!pip3 install tensorflow==1.15.0
!pip3 install h5py==2.10.0

# Most Recent One 
!pip install git+https://github.com/rcmalli/keras-vggface.git

!pip show keras-vggface #Check version

from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing import image

"""Multitask Cascaded Convolutional Networks"""

#from mtcnn import MTCNN
!pip install mtcnn
from mtcnn.mtcnn import MTCNN

"""# Data Loading and Preparation

##Obtain data
Obtain the full test set for VGGFace2
Get the metadata (this also creates the folder we want specific to the dataset):
"""

from google_drive_downloader import GoogleDriveDownloader as gdd

#download list of test files

"""Authenticate and create the PyDrive client.

"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

"""Load a file instance by ID and download to local system
"""

#create google drive file instance w file id
#file1 = drive.CreateFile({'id':''})

# Download file to colab local sys at this path
file1.GetContentFile('./vggface2/vggface2_test.tar.gz')

#might have to run a second after previous block because delay
!tar -xzvf ./vggface2/vggface2_test.tar.gz -C ./vggface2

#!rm ./vggface2/vggface2_test.tar.gz #if u need to save space

"""##Precursor Task: Detect Faces using a MTCNN

Detect faces (a separate task) in order to discern where the face is located in the image using a MTCNN. Hyperparameters used are the ones reference this [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-vpDPuJX3AhVvAp0JHUBmBkUQFnoECA0QAQ&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1604.02878&usg=AOvVaw2zP7sM4-pY8EPR1e90av6f)

"""

mtcnn_detector_default = MTCNN()

"""upload A csv of the studies's crop dimensions to use if extract_face fails. Create a dictionary for dimesnions. """

import pandas as pd

#create google drive file instance w file id
#file2 = drive.CreateFile({'id':''})

# Download file to colab local sys at this path
file2.GetContentFile('/content/vggface2/loose_bb_test.csv') 

#get csv and turn into dict
crop_dim_study_test = pd.read_csv('/content/vggface2/loose_bb_test.csv', header= 0, delimiter=",")
study_crop_dim_dict = crop_dim_study_test.set_index('NAME_ID').T.to_dict('list')

#utility method for retrieving filename from filepath 
def get_filename_from_path(filepath, fileTypeIncluded=False):
  pathSplits = filepath.split('/')
  filename_full = pathSplits[-2] +'/'+ pathSplits[-1]
  if fileTypeIncluded is True:
    return filename_full
  fileSplits = filename_full.split('.')
  return fileSplits[0]

!pip install keras_vggface
!pip3 install --upgrade keras_preprocessing

from keras_vggface.utils import preprocess_input
from keras.preprocessing import image

# extract a single face from a given photograph
def extract_face(filepath, face_detector=mtcnn_detector_default, required_size=(224, 224), study_crop_dim = False):
	# load image from file
	pixels = pyplot.imread(filepath)

	if study_crop_dim is False:
		# get the detector, using default weights
		detector = face_detector
		# detect faces in the image
		results = detector.detect_faces(pixels)
		# extract the bounding box from the first face

		#Quick fix is to load from crop dimensions given by study's dataset if no face detected
		if len(results) != 0:
			x1, y1, width, height = results[0]['box']
			x2, y2 = x1 + width, y1 + height
			# extract the face
			face = pixels[y1:y2, x1:x2]
		else: #remain uncropped
			study_crop_dim = True

	if study_crop_dim is True:
		#print("Getting from file dim")
		filename = get_filename_from_path(filepath)
		x1, y1, width, height = study_crop_dim_dict[filename] #dict returns list(X, Y,W, H)
		#running into issues with out of bound cropping
		x1 = 0 if x1 < 0 else x1
		y1 = 0 if y1 < 0 else y1
		width = 0 if width < 0 else width
		height = 0 if height < 0 else height

		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]

	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
 
	return face_array

"""Test on a sample:

Now this can be used on whatever dataset, See a sample of a cropped face:

"""

# load the photo and extract the face
face_crop = extract_face('/content/vggface2/test/n000001/0001_01.jpg', study_crop_dim=False)

# plot the extracted face
pyplot.imshow(face_crop)

# show the plot
pyplot.show()

"""#Models

Pretraining weights (the weights obtained from training on vggface2) will be downloaded when the object is created. Note the model expects input that is scaled color images of extracted face,s with the shape of 244×244. The output will be a class prediction of the number of people the model was trained on (for facial recognition/identification if the softmax layer is used) or a facial emebdding if the top (last) layer is removed< and this  ca be changed in the constructor.

Note dimensions(? = rows =N):

Inputs: [<tf.Tensor 'input_1:0' shape=(?, 224, 224, 3) dtype=float32>]

Outputs: [<tf.Tensor 'classifier/Softmax:0' shape=(?, 8631) dtype=float32>]

##Resnet50 Architecture
"""

# by default, output probs (identification)
model_resnet50_ident = VGGFace(model='resnet50')
print('Inputs: %s' % model_resnet50_ident.inputs)
print('Outputs: %s' % model_resnet50_ident.outputs)

#remove top to output facial embeddings (verification)
model_resnet50_verifi = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='max')

print('Inputs: %s' % model_resnet50_ident.inputs)
print('Outputs: %s' % model_resnet50_ident.outputs)

"""##Senet50 Architecture"""

# by default, output probs (identification)
model_senet50_ident = VGGFace(model='senet50')

#remove top to output facial embeddings (verification)
model_senet50_verifi = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='max')

"""#Changing Model Parameters

The following may be used to alter the implementation of the model using Keras, as outlined [here](https://github.com/rcmalli/keras-vggface). Note no parameters were changed for experiment 1.

## Convolution Features
"""

# Convolution Features
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='max')

# predict.

vgg_model = VGGFace()
print(vgg_model.summary())

#layer_name = 'flatten' 
layer_name = 'pool4' 
 
vgg_model = VGGFace() # pooling: None, avg or max
out = vgg_model.get_layer(layer_name).output
vgg_model_new = Model(vgg_model.input, out)

"""## Specific Layer Features"""

# Layer Features

#layer_name = 'pool1' 
layer_name = 'fc6/relu' 
vgg_model = VGGFace() # pooling: None, avg or max
out = vgg_model.get_layer(layer_name).output
vgg_model_new = Model(vgg_model.input, out)

"""##Finetuning"""

nb_class = 2

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool4').output
x = Flatten(name='flatten')(last_layer)
out = Dense(nb_class, activation='relu', name='classifier')(x)
custom_vgg_model = Model(vgg_model.input, out)

from PIL import Image
from keras_vggface.utils import preprocess_input

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filepath_list, model, study_crop_dim = False, pre_pro_version=2):
  # extract faces
  faces = [extract_face(f, study_crop_dim = study_crop_dim) for f in filepath_list]  
  # convert into an array of samples
  samples = np.asarray(faces, 'float32')
  # prepare the face for the model, e.g. center pixels
  samples = preprocess_input(samples, version=pre_pro_version)
  # perform prediction
  yhat = model.predict(samples)
  return yhat

from google_drive_downloader import GoogleDriveDownloader as gdd

#method reads all lines from metadata file and store in list    
file = open('./Experiment1/age_exp_imglist.txt', 'r')
age_image_path_list = file.read().splitlines()

file.close()
print("Number of files specified: ", len(age_image_path_list))

#add begining of absolute path to list
prefix = '/content/vggface2/test/'
for i in range(len(age_image_path_list)):
  age_image_path_list[i] = prefix + age_image_path_list[i]

#predict using default resnet50 architecture for verification 
yhat_age_resnet = get_embeddings(age_image_path_list, model=model_resnet50_verifi, study_crop_dim=True)

print(yhat_age_resnet.shape) #with omitted subject: 1980, 2048

#predict using default resnet50 architecture for verification 
yhat_age_senet = get_embeddings(age_image_path_list, model =model_senet50_verifi, study_crop_dim=True)

print(yhat_age_senet.shape) #with omitted subject: 1980, 2048

from scipy.spatial import distance

# mean used to calculate a template score
def template_score(template):
  template_score = np.mean(template, axis=0)
  return template_score

#compute cosine similarity of one template to another
def template_distance_score(template1, template2):
  #Compute the Cosine distance between 1-D arrays
  score = 1 - distance.cosine(template1, template2) #want similarity so 1-x
  return score

#compute the 2x2 matrix for each subject (we'll store it flatted for later)
#subject should be  (20, 2048) array
def create_sim_matrix(subject):
  #splice yhat subset representing subject
  template1, template2, template3, template4  = subject[0:5], subject[5:10], subject[10:15], subject[15:20] #so template dimensions should be (5, 2048)

  #get template scores
  t1_score = template_score(template1)  #young1
  t2_score = template_score(template2)  #young2
  t3_score = template_score(template3)  #mature1
  t4_score = template_score(template4)  #mature2

  #get the similarity matrix
  sim_matrix = np.zeros((4))

  sim_matrix[0] = template_distance_score(t1_score, t2_score) #young1, young2
  sim_matrix[1] = template_distance_score(t1_score, t3_score) #young1, mature1
  sim_matrix[2] = template_distance_score(t4_score, t2_score) #mature, young
  sim_matrix[3] = template_distance_score(t3_score, t4_score) #mature, mature

  return sim_matrix

def get_comparisons(facial_embeddings): #facial embeddigns will be yhat
  j = 20
  subject_num = 0

  all_sim_matrices = np.zeros((int(facial_embeddings.shape[0]/20), 4), dtype=float)

  for i in range(0, facial_embeddings.shape[0], 20): 
    subject = facial_embeddings[i:j]
    sim_matrix = create_sim_matrix(subject)
    all_sim_matrices[subject_num] = sim_matrix
    subject_num = subject_num + 1
    j += 20
    
  print("Number of subjects")
  print(subject_num)
  return all_sim_matrices #will be in 1x4 format instead of 2x2

#Resnet results

all_sim_matrices_age_res = get_comparisons(yhat_age_resnet)
print("All Resnet similiarity matrices:")
#print(all_sim_matrices_age_res)
print(all_sim_matrices_age_res.shape)

#take mean across all subjects
mean_sim_matrices_age_res = np.mean(all_sim_matrices_age_res, axis = 0) #should be 4x1 matrix

print("Resnet Similiarty Matrix Means:")
print(mean_sim_matrices_age_res)


#SENet Results
all_sim_matrices_age_se = get_comparisons(yhat_age_senet)
print("All SENet similiarity matrices:")
#print(all_sim_matrices_age_se)
print(all_sim_matrices_age_se.shape)

#take mean across all subjects
mean_sim_matrices_age_se = np.mean(all_sim_matrices_age_se, axis = 0) #should be 4x1 matrix

print("SEnet Similiarty Matrix Means:")
print(mean_sim_matrices_age_se)