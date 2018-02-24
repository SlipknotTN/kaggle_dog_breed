# Kaggle playground competition "Dog breed identification" scripts

[Dog breed identification](https://www.kaggle.com/c/dog-breed-identification/leaderboard) is a Kaggle competition on fine grained image classification task.
The available dataset has a low number of images for each class (120 classes, around 100 images for each one).

[Interesting Kaggle kernel](https://www.kaggle.com/gaborfodor/dog-breed-pretrained-keras-models-lb-0-3) that shows an easy an effective way to build a model ensemble.

In my work I used small deep learning models, an Nvidia CUDA laptop is able to train them in acceptable times.
I think that better results could be achieved only by using bigger models and keeping the same training pipeline.

Tensorflow models are imported from TF slim format (NasNet Mobile and Mobilenet) or converted from Caffe (SqueezeNet).
[Here](https://github.com/SlipknotTN/Dogs-Vs-Cats-Playground/tree/master/deep_learning/tensorflow) I explain how to obtain the correct format of these Tensorflow model.

## Single models results

Some results, best one for each reference configuration.

| Base Model       | Framework | Architecture | Hyperparameters | Kaggle Score |
|-------|------------|-----------------|---------------|------------------|
| MobileNet   | Keras     | Pretrained base model + mobilenet top classifier |   SGD LR 0.001 fixed, Momentum 0.9   |  0.89499 |
| MobileNet   | Tensorflow| Pretrained base model + mobilenet top classifier |   Adam LR 0.001   |  0.96290 |
| NasNet Mobile | Tensorflow | Pretrained base model + nasnet top classifier |   Adam LR 0.001   |  0.71314 |
| MobileNet | Tensorflow + Scikit | Bottleneck features + logistic regression | L-BFGS Solver | 0.66534 |
| NasNet Mobile | Tensorflow + Scikit | Bottleneck features + logistic regression | L-BFGS Solver | **0.65946** |
| MobileNet | Tensorflow + Scikit | Bottleneck features + logistic regression | L-BFGS Solver | 2.04384 |

Multinomial logistic regression takes as input bottleneck features from pretrained deep learning models.
Bottleneck features are exported with Tensorflow framework and saved to file as npy array.
Extracted features are read flattened as 1D array.

Selected layers as bottleneck (Tensorflow tensor exact names):
- MobileNet: MobilenetV1/Logits/Dropout_1b/Identity with shape (batch, 1, 1, 1024)
- NasNet Mobile: final_layer/dropout/Identity with shape (batch, 1056)
- SqueezeNet v1.1: fire9_concat with shape (batch, 14, 14, 512) and then applied a global average to obtain (batch, 1, 1, 512).

SqueezeNet fire9_concat layer has a big shape size that makes impossible to train a logistic regression on it.
I tried to apply a global average pooling to reduce the size, but, as we can see from the results, it is not very effective.
In the full SqueezeNet model the global average is applied only after the last convolution trained on the desired classes.


## Ensemble of models results

Same features bottleneck extracted for the single models predictions. The logistic regression classifier is trained using as input
 a 1D array made with the concatenated features.

| Models       | Framework | Architecture | Hyperparameters | Kaggle Score |
|-------|------------|-----------------|---------------|------------------|
| MobileNet + NasNet Mobile  | Tensorflow + Scikit     | Concatenated bottlenecks features + logistic regression |   L-BFGS Solver | **0.50718** |
| MobileNet + NasNet Mobile + SqueezeNet | Tensorflow + Scikit     | Concatenated bottlenecks features + logistic regression |   L-BFGS Solver | 0.79925 |
