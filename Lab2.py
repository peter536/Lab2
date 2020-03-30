import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
tf.compat.v1.set_random_seed(1618)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Comment out whichever set you don't want to use

#CONTENT_IMG_PATH = "/Users/alex_p/Desktop/CS390NIP/Lab2/Shreveport.png"            
#STYLE_IMG_PATH = "/Users/alex_p/Desktop/CS390NIP/Lab2/mushroom.png"            

CONTENT_IMG_PATH = "/Users/alex_p/Desktop/CS390NIP/Lab2/dog.png"
STYLE_IMG_PATH = "/Users/alex_p/Desktop/CS390NIP/Lab2/pond.png"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.025    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 5



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
### This function was written with help from https://keras.io/examples/neural_style_transfer/ ###
def deprocessImage(img):
    if K.image_data_format() == "channels_first":
        img = img.reshape((3, CONTENT_IMG_H, CONTENT_IMG_W))
        img = img.transpose((1, 2, 0))
    else:
        img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # Remove zero-center by mean pixel
    img[:,:,0] += 103.939
    img[:,:,1] += 116.779
    img[:,:,2] += 123.68
    # 'BGR'->'RGB'
    img = img[:,:,::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img

### This function was written with help from the class slides ###
def gramMatrix(x):
    if K.image_data_format() == "channels_first":  
        features = K.flatten(x)
    else: 
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

### This function was written with help from https://keras.io/examples/neural_style_transfer/ ###
def styleLoss(style, gen):
    #assert K.ndim(style) == 3
    #assert K.ndim(gen) == 3
    S = gramMatrix(style)
    G = gramMatrix(gen)
    channels = 3 ### this is the numFilters from the class slides
    size = STYLE_IMG_H * STYLE_IMG_W
    return K.sum(K.square(S - G)) / (4.0 * (channels ** 2) * (size ** 2))
    #return None   #TODO: implement.##############################


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    #assert K.ndim(x)== 4
    if K.image_data_format() == "channels_first":
        a = K.square(
            x[:, :, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1] - x[:, :, 1:, :CONTENT_IMG_W - 1])
        b = K.square(
            x[:, :, :STYLE_IMG_H - 1, :STYLE_IMG_W - 1] - x[:, :, STYLE_IMG_H - 1, 1:])
    else:
        a = K.square(
            x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
        b = K.square(
            x[:, :STYLE_IMG_H - 1, :STYLE_IMG_W - 1, :] - x[:, :STYLE_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
    #return None   #TODO: implement.################################





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor) ######
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    
    loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)

    print("   Calculating style loss.")

    for layerName in styleLayerNames:
        layer_features = outputDict[layerName]
        style_reference_features = layer_features[1, :, :, :]
        gen_features = layer_features[2, :, :, :]    
        loss += (STYLE_WEIGHT / len(styleLayerNames)) * styleLoss(style_reference_features, gen_features)
    
    loss += TOTAL_WEIGHT * totalLoss(genTensor)
    # TODO: Setup gradients or use K.gradients().###########################
    grads = K.gradients(loss, genTensor)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    
    f_outputs = K.function([genTensor], outputs)

    def eval_loss(x):
        if K.image_data_format() == "channels_first":
            x = x.reshape((1, 3, CONTENT_IMG_H, CONTENT_IMG_W))
        else:
            x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        outs = f_outputs([x])
        loss_val = outs[0]
        return loss_val

    def eval_grads(x):
        if K.image_data_format() == "channels_first":
            x = x.reshape((1, 3, CONTENT_IMG_H, CONTENT_IMG_W))
        else:
            x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        outs = f_outputs([x])
        if len(outs[1:]) == 1:
            grad_vals = outs[1].flatten().astype("float64")
        else:
            grad_vals = np.array(outs[1:]).flatten().astype("float64")
        return grad_vals
        
    loadedImg = load_img(CONTENT_IMG_PATH)
    x = preprocessData((loadedImg, CONTENT_IMG_H, CONTENT_IMG_W))

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.#######################
        x, min, info = fmin_l_bfgs_b(eval_loss, x.flatten(), fprime=eval_grads, maxfun=20)
        print("      Loss: %f." % min)
        img = deprocessImage(x.copy())
        saveFile = "/Users/alex_p/Desktop/CS390NIP/Lab2/" + "transfer_round_%d.png" % i
        imsave(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")
    


#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
