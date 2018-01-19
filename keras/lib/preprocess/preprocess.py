from keras.applications import imagenet_utils
from keras.applications import mobilenet


def dummyPreprocessInput(image):
    image -= 127.5
    return image


def getPreprocessFunction(preprocessType):

    if preprocessType == "dummy":
        return dummyPreprocessInput
    elif preprocessType == "mobilenet":
        return mobilenet.preprocess_input
    elif preprocessType == "imagenet":
        return imagenet_utils.preprocess_input
    else:
        raise Exception(preprocessType + " not supported")
