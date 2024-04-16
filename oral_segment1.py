from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import seaborn as sn
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob as gb
from tqdm import tqdm
import pandas as pd
from skimage.io import imread
import tensorflow as tf
from PIL import Image,ExifTags
            
from keras import backend as K
import segmentation_models as sm

#Scikitlearn
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix,f1_score

print("tensorflow version : ",tf.__version__)

import sys
sys.path.append('../')
from mylib import mylib as ml
from mylib import mymetric as mt

#Segmentation model
sm.set_framework('tf.keras')
sm.framework()
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # this MUST come before any tf call 

#GPU set memory growth for tensorflow library
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
    print("Success")
except:
  # Invalid device or cannot modify virtual devices once initialized.
    print("GOT EXCEPTION")    
    pass


'''

Loading Image and Ground Truth

'''

#Path of images and labels 
data_type = "Full images" # full image of image (Original oral images) or Croping Mouth images (Croping Mouth using mouth detection model)
img_path = "../Dataset/trainning2/images"
gt_path = "../Dataset/trainning2/labels"


images_dir = gb.glob(img_path+"/*")
labels_dir = gb.glob(gt_path+"/*")
images_dir.sort()
labels_dir.sort()
images_dir
print("Image has 8 dataset")
print(len(images_dir),len(labels_dir))

#Load list of images each class
images_lst = []
for i in images_dir:
    fname_lst = gb.glob(i+"/*")
    fname_lst.sort()
    images_lst.append(fname_lst)
images_lst.sort()
    
labels_lst = []
for i in labels_dir:
    fname_lst = gb.glob(i+"/*")
    fname_lst.sort()
    labels_lst.append(fname_lst)
labels_lst.sort()
print(len(images_lst),len(labels_lst))
for i in range(len(images_lst)):
    print(len(images_lst[i]),len(labels_lst[i]))

# Selection Model Achitecture and setting
data = {}
data["ModelName"] = "segment_oral_multi_organ_test_1"  # Your model name
data["ModelACTT"] = "FPN" # model
data["BackBone"] = "efficientnetb5" #efficientnetb5 #mobilenetv2 efficientnetb5 resnet50
data["Class"] = 7+1
data["Epoch"] = 5
data["BatchSize"] = 16
data["ImgSize"] = (512,512)



#set Path to save all result files
save_model_path = "../Oral_segment_result/"
if os.path.isdir(save_model_path+data["ModelName"]) == False:
    os.mkdir(save_model_path+data["ModelName"])
    ml.dict_2_txt(data,save_model_path+"data.txt") # save data detail
else:
    print("Already have this model name. This model will replace Your old model")
save_model_path = save_model_path+data["ModelName"]+"/"



import random #SetSeed for random image
data['randomstate_splitdata'] = 804
random.seed(data['randomstate_splitdata'])

#split train test valid by stratified spliting
Xtrain_n,Ytrain_n,Xvalid_n,Yvalid_n,Xtest_n,Ytest_n = ml.Split_Dataset(images_lst,labels_lst,valid_size=0.1,test_size=0.2,seed=data['randomstate_splitdata'])
print(len(Xtrain_n),len(Ytrain_n),len(Xvalid_n),len(Yvalid_n),len(Xtest_n),len(Ytest_n))


#Resize Normal form
Xtrain,Ytrain = [],[]
Xvalid, Yvalid = [] ,[]
Xtest,Ytest = [],[]

print("Load Train set")
for k,(m,n) in enumerate(zip(Xtrain_n,Ytrain_n)):
    print(" "*20,end="\r")
    print(k+1,"/",len(Xtrain_n),end="\r")
    im = cv2.imread(m)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,data['ImgSize'])
    lb = cv2.imread(n, 0) 
    lb = cv2.resize(lb,data['ImgSize'])
    Xtrain.append(im)
    Ytrain.append(lb)
print("Loaded Trainning Set")
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
print(Xtrain.shape,Ytrain.shape)

print("Load validation set")
for k,(m,n) in enumerate(zip(Xvalid_n,Yvalid_n)):
    print(" "*20,end="\r")
    print(k+1,"/",len(Xvalid_n),end="\r")
    im = cv2.imread(m)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,data['ImgSize'])
    lb = cv2.imread(n, 0) 
    lb = cv2.resize(lb,data['ImgSize'])
    Xvalid.append(im)
    Yvalid.append(lb)
print("Loaded validation Set")
Xvalid = np.array(Xvalid)
Yvalid = np.array(Yvalid)
print(Xvalid.shape,Yvalid.shape)


print("Load test set")
for k,(m,n) in enumerate(zip(Xtest_n,Ytest_n)):
    print(" "*20,end="\r")
    print(k+1,"/",len(Xtest_n),end="\r")
    im = cv2.imread(m)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,data['ImgSize'])
    lb = cv2.imread(n, 0) 
    lb = cv2.resize(lb,data['ImgSize'])
    Xtest.append(im)
    Ytest.append(lb)
print("Loaded Test Set")
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)
print(Xtest.shape,Ytest.shape)

#Categorical to one hot vector
Ytrain = to_categorical(Ytrain,num_classes=np.max(Ytrain)+1)
Yvalid = to_categorical(Yvalid,num_classes=np.max(Yvalid)+1)
Ytest = to_categorical(Ytest,num_classes=np.max(Ytest)+1)

Xtrain = Xtrain.astype(np.float32)
Xvalid = Xvalid.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Ytrain = Ytrain.astype(np.float32)
Yvalid = Yvalid.astype(np.float32)
Ytest = Ytest.astype(np.float32)


'''

Initial Train-time Augment image Function 

'''

#Create function for train-time augmentation
# we apply ImageDatagenerator for image segmentation task

def trainGenerator(Xtrain, Ytrain,batchsize=8,seed=804):
    
    image_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='constant'
    )
    mask_datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='constant'
    )
    
    image_generator = image_datagen.flow(
        Xtrain,
        batch_size=batchsize,
        seed=seed
    )
    
    mask_generator = mask_datagen.flow(
        Ytrain,
        batch_size=batchsize,
        seed=seed
    )
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        yield (img, mask)
        
def validGenerator(X, Y,batchsize=8,seed=804):
    
    image_datagen = ImageDataGenerator(
        rescale=1./255,
    )
    mask_datagen = ImageDataGenerator(
    )
    
    image_generator = image_datagen.flow(
        X,
        batch_size=batchsize,
        seed=seed
    )
    
    mask_generator = mask_datagen.flow(
        Y,
        batch_size=batchsize,
        seed=seed
    )
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        yield (img, mask)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = trainGenerator(Xtrain,Ytrain,batchsize=data['BatchSize'],seed=data['randomstate_splitdata'])
valid_generator = validGenerator(Xvalid,Yvalid,batchsize=data['BatchSize'],seed=data['randomstate_splitdata'])

# Augment Detail
data["Augment"] = '''
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant'
                '''


'''

Initial Optimize Algorithm and Setting Params.

'''
#initial Optimizor
data["Optimizer"] = "Adam"
data["LearningRate"] = 0.0001
data["Beta1"] = 0.9
data["Beta2"] =  0.999

from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau ,ModelCheckpoint ,TensorBoard
optimizer_adam=Adam(learning_rate=data["LearningRate"],beta_1=data["Beta1"],beta_2=data["Beta2"])


data["EarlyStop"] = 20 #Number of Epoch to stop traning mdoel when validation set loss can't reduce
data['lr_patience'] = 10 #Number of Epoch to reduce learning rate when training set loss can't reduce
data['factor'] = 0.1 # rate for reduce learning rate by: new_lr =  old_lr * factor
data["min_lr"] = 0.0 # Minimum learning rate can reduce
EarlyStop=EarlyStopping(patience=data["EarlyStop"],restore_best_weights=True,verbose=1) #when train and it can't get any better than this
Reduce_LR=ReduceLROnPlateau(monitor='loss',verbose=1,factor=data['factor'],patience=data['lr_patience'],min_lr=data["min_lr"])
model_check=ModelCheckpoint(save_model_path+data["ModelName"]+'_best.h5',monitor='val_loss',verbose=1,save_best_only=True)
tensorbord=TensorBoard(log_dir='logs')

#Packing Training function 
if Reduce_LR and EarlyStop:
    callback=[EarlyStop,Reduce_LR,model_check,tensorbord]
    data['Reduce_LR'] = 'True'
    data['Earlystop'] = 'True'
    print("Activate Reduce_LR and EarlyStop")
elif Reduce_LR:
    callback=[Reduce_LR,model_check,tensorbord]
    data['Reduce_LR'] = 'True'
    data['Earlystop'] = 'False'
    print("Activate Reduce_LR")
elif EarlyStop:
    callback=[EarlyStop,model_check,tensorbord]
    data['Reduce_LR'] = 'False'
    data['Earlystop'] = 'True'
    print("Activate EarlyStop")
else:
    callback=[model_check,tensorbord]
    print("No add-ons Function")



#focal jaccard loss
loss = sm.losses.categorical_focal_jaccard_loss

# Define the strategy (Multi gpu process) 
# strategy = tf.distribute.MirroredStrategy() # This line can use on multi gpu computer

#Load model
#Segmentation model
# with strategy.scope():
if data["ModelACTT"] == "UNET":
    model = sm.Unet(data["BackBone"], classes=data["Class"],input_shape=(data["ImgSize"][1], data["ImgSize"][0], 3), activation='softmax')
elif data["ModelACTT"] == "FPN":
    model = sm.FPN(data["BackBone"], classes=data["Class"],input_shape=(data["ImgSize"][1], data["ImgSize"][0], 3), activation='softmax')
elif data["ModelACTT"] == "LINKNET":
    model = sm.Linknet(data["BackBone"], classes=data["Class"],input_shape=(data["ImgSize"][1], data["ImgSize"][0], 3), activation='softmax')
elif data["ModelACTT"] == "DEEPLABV3+":
    if data["BackBone"] == "efficientnetb5":
        model = ml.DeeplabV3Plus_effb5(image_size = (data["ImgSize"][0],data["ImgSize"][1],3), num_classes=data["Class"],activation="softmax")
    elif data['BackBone'] == "resnet50":
        model = ml.DeeplabV3Plus(image_size = (data["ImgSize"][0],data["ImgSize"][1],3), num_classes=data["Class"],activation="softmax")
    elif data['BackBone'] == "mobilenetv2":
        model = ml.DeeplabV3Plus_MobileNetV2(image_size = (data["ImgSize"][0],data["ImgSize"][1],3), num_classes=data["Class"],activation="softmax")
model.compile(optimizer=optimizer_adam,loss=loss,metrics=["accuracy",loss])


num_class = data['Class']
data['Model_loss'] = 'sm.losses.categorical_focal_jaccard_loss'


with tf.device('/gpu'):
    history = model.fit(train_generator, 
                    epochs=data["Epoch"], # one forward/backward pass of training data
                    steps_per_epoch=Xtrain.shape[0]//data["BatchSize"], # number of images comprising of one epoch
                    validation_data=valid_generator, # data for validation
                    validation_steps=Xvalid.shape[0]//data["BatchSize"],
                    callbacks=callback, 
                    verbose=1
                    # workers=1
                    # use_multiprocessing=False         
                    )

#Save history
hist_df = pd.DataFrame(history.history) 
model.save(save_model_path+data["ModelName"]+".h5")
hist_df.to_csv(save_model_path+data["ModelName"]+".csv",index=False)


xtest = test_datagen.flow(Xtest, batch_size=len(Xtest),shuffle=False)
xtest = xtest[0]


# Plot model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.savefig(save_model_path+data["ModelName"]+'.png')
plt.close()

lr = history.history['lr']
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.title('Model Learning rate Controller')
plt.plot(lr)
plt.savefig(save_model_path+'learning rate.png')
plt.close()

#loop add my program
lst_y = []
with tf.device('/gpu'):
    for i in xtest:
        y_pred=model.predict(np.array([i]))
        lst_y.append(y_pred[0])
    y_pred = np.array(lst_y)
    print("Predicted test")


y_pred_num = np.argmax(y_pred, axis=3, out=None)
y_test_num = np.argmax(Ytest, axis=3, out=None)
x_test = Xtest.astype(int)




# Finding F1-Score 
def F1_score(list_of_groundtruth,list_of_predict ,num_classes=None):
    
'''

    Inputs : list_of_groundtruth = numpy.array (W,H) ex.(512,512)
             list_predict        = numpy.array (W,H)
             num_classes         = int
    
    Outputs : dictionary of 
                data['f1_avg'] = f1_scores                                # f1_scores : F1-Score seperate each class
                data['f1_macro'] = macro_f1                               # macro_f1 : sum(f1_scores) / len(f1_socres)                          
                data['weighted_precisions'] = weighted_average_precisions # weighted_average_precisions = sum( precisions[class_i] * number_of_pixels[class_i] ) / sum(number_of_pixels[all_class])
                data['weighted_recalls'] = weighted_average_recalls       # weighted_average_recalls    = sum( recalls[class_i] * number_of_pixels[class_i] ) / sum(number_of_pixels[all_class])
                data['f1_weighted'] = weighted_f1                         # weighted_f1 = harmonic mean of weighted_precisions and weighted recalls

'''

    if num_classes == None:
        num_classes = len(np.unique(list_of_groundtruth[0]))
    precisions,recalls = [0]*num_classes ,[0]*num_classes
    tps,fps,fns = [0]*num_classes ,[0]*num_classes,[0]*num_classes
    f1_scores = [0]*num_classes
    pixel_counts = [0]*num_classes
    
    for i,(y,p) in enumerate(zip(list_of_groundtruth[:],list_of_predict[:])):
        print(" "*20,end="\r")
        print(i+1,"/",len(list_of_groundtruth),end="\r")
        ground_truth , predicted_labels = y.flatten(),p.flatten()

        for class_id in range(num_classes):
            true_positive = np.sum((ground_truth == class_id) & (predicted_labels == class_id))
            false_positive = np.sum((ground_truth != class_id) & (predicted_labels == class_id))
            false_negative = np.sum((ground_truth == class_id) & (predicted_labels != class_id))

            tps[class_id] += true_positive
            fps[class_id] += false_positive
            fns[class_id] += false_negative

        for k in range(num_classes):
            pixel_counts[k] += np.sum(ground_truth == k)

    for class_id in range(num_classes):
    #         Calculate precision and recall for the current class
            precisions[class_id] = tps[class_id] / (tps[class_id] + fps[class_id]) if (tps[class_id] + fps[class_id]) > 0 else 0
            recalls[class_id] = tps[class_id] / (tps[class_id] + fns[class_id]) if (tps[class_id] + fns[class_id]) > 0 else 0

            # Calculate F1-score for the current class
            if precisions[class_id] + recalls[class_id] == 0:
                f1_scores[class_id] = 0  # Handle division by zero
            else:
                f1_scores[class_id] = 2 * (precisions[class_id] * recalls[class_id]) / (precisions[class_id] + recalls[class_id])


    # Calculate the weighted Jaccard scores for each class
    weighted_precisions = [score * weight for score, weight in zip(precisions, pixel_counts)]
    weighted_recalls = [score * weight for score, weight in zip(recalls, pixel_counts)]

    # Calculate the total weighted Jaccard score
    total_weighted_precisions= sum(weighted_precisions)
    total_weighted_recalls= sum(weighted_recalls)
    # Calculate the sum of the weights (total pixels)
    total_weight = sum(pixel_counts)
    # Calculate the weighted average Jaccard score
    weighted_average_precisions = total_weighted_precisions / total_weight
    weighted_average_recalls = total_weighted_recalls / total_weight

    # Calculate F1-score for the current class
    if weighted_average_precisions + weighted_average_recalls == 0:
        weighted_f1 = 0  # Handle division by zero
    else:
        weighted_f1 = 2 * (weighted_average_precisions * weighted_average_recalls) / (weighted_average_precisions + weighted_average_recalls)
    macro_f1= sum(f1_scores)/len(f1_scores)
    data = {}
    data['f1_avg'] = f1_scores
    data['f1_macro'] = macro_f1
    data['weighted_precisions'] = weighted_average_precisions
    data['weighted_recalls'] = weighted_average_recalls
    data['f1_weighted'] = weighted_f1
    
    return data
    
num_classes = data['Class']
res = F1_score(y_test_num,y_pred_num,num_classes=num_classes)
data['f1_avg'] = res['f1_avg']
data['f1_macro'] = res['f1_macro']
data['weighted_precisions'] = res['weighted_precisions']
data['weighted_recalls'] = res['weighted_recalls']
data['f1_weighted'] = res['f1_weighted']



all_cf = np.zeros((num_classes,num_classes))
all_cf = all_cf.astype(np.int64)
print(all_cf)
num_classes = 8
for i,(y,p) in enumerate(zip(y_test_num[:],y_pred_num[:])): # Y_pred_fullsize for cropmouth data
    print(" "*100,end="\r")
    print(i+1,"/",len(y_test_num),end="\r")
    ground_truth,predicted = y.astype(np.int64).flatten(),p.astype(np.int64).flatten()
    cf = np.bincount(ground_truth * int(num_classes) + predicted, minlength=num_classes**2).reshape(num_classes, num_classes)
    all_cf += cf
    
data['cf'] = all_cf

# plt.rcParams['font.family'] = 'tahoma'  
new_arr = np.zeros(all_cf.shape)
for i,v in enumerate(all_cf):
    new_arr[i] = (all_cf[i]/np.sum(all_cf[i]))
round_new_arr = np.round(new_arr,2)
# sn.set()
lst_cls = ["Background","Other Inside","Bucca Mucosa","Floor of Mouth","Gums","Lips","Tongue","Tooth"]
# lst_cls = ['พื้นหลัง','บริเวณอื่นในช่องปาก','กระพุ้งแก้ม','พื้นปาก','เหงือก','ริมฝีปาก','ลิ้น','ฟัน']
df_cm = pd.DataFrame(round_new_arr, index = [i for i in lst_cls], columns = [i for i in lst_cls])
plt.figure(figsize = (12,10))
sn.heatmap(df_cm,vmin = 0,fmt = '.0%', annot=True, square=1,cmap="Blues",annot_kws={"fontsize":14})
plt.xticks(rotation = 35,fontsize=14)
plt.yticks(rotation = 0,fontsize=14)
plt.xlabel("Predicted",fontweight='bold',fontsize=16)
plt.ylabel("True",fontweight='bold',fontsize=16)
plt.savefig(save_model_path+"/"+"confuse_matrix.png", bbox_inches = "tight") # full images
plt.close()


ml.dict_2_txt(data,save_model_path+"/data.txt")




from matplotlib import cm
def map_label2image(img,label,tresh=128):
    '''
    Function for map mask color to draw on RGB image
    image type as numpy array (data range'0-255' or '0-1')
    label type as numpy array (data uint8 expect class of pixel)
    tresh range of mask transparent (type int '0-255')
    '''
    
    if np.max(img) <= 1.0:
        img = img*255.0
    imp = Image.fromarray(img.astype(np.uint8))
    t = Image.fromarray(np.uint8(cm.tab20(label)*255)) #tab10
    mask = np.where(label>0,tresh,0)
    mask = Image.fromarray(mask.astype(np.uint8))
#     return np.array(imp.paste(t,(0, 0),mask=mask))
    imp.paste(t,(0, 0),mask=mask)
    return np.array(imp)


if os.path.isdir(save_model_path) == False:
    os.mkdir(save_model_path)
addition  = "" # for retest or somthing else

save_imgs = save_model_path +addition+ "images_result/"
if os.path.isdir(save_imgs) == False:
    os.mkdir(save_imgs)
print("image save path :",save_imgs)


save_overlay_512 = save_imgs + "overlay_512/"
if os.path.isdir(save_overlay_512) == False:
    os.mkdir(save_overlay_512)
print("overlay_512 save path :",save_overlay_512)

save_pred_mask = save_imgs + "original_pred_mask/"
if os.path.isdir(save_pred_mask) == False:
    os.mkdir(save_pred_mask)
print("pred mask save path :",save_pred_mask)


for i in range(len(y_pred_num)):
    print("  "*100,end="\r")
    print(i+1,"/",len(y_pred_num),end="\r")
    cv2.imwrite(save_overlay_512+Xtest_n[i].split("/")[-1],cv2.cvtColor(map_label2image(Xtest[i].astype(np.uint8),y_pred_num[i].astype(np.uint8)), cv2.COLOR_RGB2BGR))

for i in range(len(y_pred_num)):
    print("  "*100,end="\r")
    print(i+1,"/",len(y_pred_num),end="\r")
    cv2.imwrite(save_pred_mask+Xtest_n[i].split("/")[-1],cv2.cvtColor(y_pred_num[i].astype(np.uint8), cv2.COLOR_RGB2BGR))