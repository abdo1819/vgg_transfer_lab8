[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/anujshah645)

# Transfer-Learning-in-keras---custom-data
for the compatible version with keras-2 and other issues you can have a look at the corrections made by people in pull request, and also this page - https://github.com/kusiwu/Transfer-Learning-in-keras---custom-data
I am not changing here as it is compatible with versions i have metioned in the tutorial

The video tutorial for Transfer learning with VGG-16 : https://www.youtube.com/watch?v=L7qjQu2ry2Q&feature=youtu.be

The video tutorial for Transfer learning with Resnet-50 : https://youtu.be/m5RjXjvAAhQ

This repository shows how we can use transfer learning in keras with the example of training a 4 class classification model using VGG-16 and Resnet-50 pre-trained weights.The vgg-16 and resnet-50 are the CNN models trained on more than a million images of 1000 different categories.

Transfer learning refers to the technique of using knowledge of one domain to another domain.i.e. a NN model trained on one dataset can be used for other dataset by fine-tuning the former network.

Definition : Given a source domain Ds and a learning task Ts, a target domain Dt and learning task Tt, transfer learning aims to help improve the learning of the the target predictive function Ft(.) in Dt using the knowledge in Ds and Ts, where Ds ≠ Dt, or Ts ≠ Tt.

A good explanation of how to use transfer learning practically is explained in http://cs231n.github.io/transfer-learning/

## When and how to fine-tune?

How do you decide what type of transfer learning you should perform on a new dataset?
This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity
to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images).
Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, 
here are some common rules of thumb for navigating the 4 major scenarios:

	New dataset is small and similar to original dataset. Since the data is small, it is not a good idea to fine-tune the ConvNet 
due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be 
relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

	New dataset is large and similar to the original dataset. Since we have more data, we can have more confidence that we won’t 
overfit if we were to try to fine-tune through the full network.

	New dataset is small but very different from the original dataset. Since the data is small, it is likely best to only train a 
linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, 
which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere 
earlier in the network.

	New dataset is large and very different from the original dataset. Since the dataset is very large, we may expect that we can 
afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a 
pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

# Trained image classification models for Keras

**THIS REPOSITORY IS DEPRECATED. USE THE MODULE `keras.applications` INSTEAD.**

Pull requests will not be reviewed nor merged. Direct any PRs to `keras.applications`. Issues are not monitored either.

----

This repository contains code for the following Keras models:

- VGG16
- VGG19
- ResNet50
- Inception v3
- CRNN for music tagging

All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at `~/.keras/keras.json`. For instance, if you have set `image_dim_ordering=tf`, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (`weights='imagenet'` argument in model constructor for all image models, `weights='msd'` for the music tagging model). Weights are automatically downloaded if necessary, and cached locally in `~/.keras/models/`.

## Examples

### Classify images

```python
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
```

### Extract features from images

```python
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer

```python
from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - please cite this paper if you use the VGG models in your work.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - please cite this paper if you use the ResNet model in your work.
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) - please cite this paper if you use the Inception v3 model in your work.
- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

Additionally, don't forget to [cite Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras) if you use these models.


## License

- All code in this repository is under the MIT license as specified by the LICENSE file.
- The ResNet50 weights are ported from the ones [released by Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- The VGG16 and VGG19 weights are ported from the ones [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).
- The Inception v3 weights are trained by ourselves and are released under the MIT license.
