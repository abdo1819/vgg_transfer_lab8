import numpy as np
from vgg16 import VGG16
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
import tensorflow as tf
import os

model_path = 'model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = VGG16(include_top=True, weights='imagenet')
    # model.save('model.keras')


print("saved")

img_path = r'data\data\cats\cat.1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

model.summary()
model.layers[-1].get_config()

#%%

# model = VGG16(weights='imagenet', include_top=False)

# model.summary()
# model.layers[-1].get_config()

# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = model.predict(x)

# #%%

# model = ResNet50(include_top=True,weights='imagenet')
# model.summary()
# model.layers[-1].get_config()
# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# #
# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds))
# ## print: [[u'n02504458', u'African_elephant']]
# #
# ##%%
# model = ResNet50(include_top=False,weights='imagenet')
# model.summary()
# model.layers[-1].get_config()
# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
