import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sie

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

train = pd.read_csv("train.csv")

# print(train.shape)
# print(train.head())

y_train = train["label"]
x_train = train.drop(["label"], axis=1)

del train

# print(y_train.value_counts()) # Shows the number of each label in the dataset.
# print(x_train.isnull().any().describe())

x_train = x_train / 255.0
x_train = x_train.values.reshape(-1, 28, 28, 1)
# Turkish
"""Bu, veriyi dört boyutlu bir tensöre dönüştürmek için kullanılır. 
İlk boyut, -1 ile ifade edilmiştir, bu da numpy tarafından otomatik 
olarak belirlenen bir boyut demektir (genellikle veri sayısı). 
İkinci ve üçüncü boyutlar, görüntü boyutlarıdır (28x28 piksel), ve 
dördüncü boyut, kanal sayısını temsil eder. Burada, kanal sayısı 1 
olarak belirlenmiştir çünkü görüntüler siyah-beyaz olduğu için tek 
bir kanala sahiptir. Renkli görüntülerde genellikle üç kanal bulunur 
(RGB)."""

# English
""""This is used to transform the data into a four-dimensional tensor. 
The first dimension is denoted by -1, which represents a dimension 
automatically inferred by numpy (often the number of data points). 
The second and third dimensions represent the image dimensions 
(28x28 pixels), and the fourth dimension represents the number of 
channels. Here, the number of channels is specified as 1 because the
images are grayscale, hence having a single channel. In colored 
images, there are typically three channels (RGB)."""

# One-Hot encoding
y_train = np_utils.to_categorical(y_train, num_classes = 10)

# print(y_train)
# print(y_train.shape)

random_seed = 6

# %10 of the training data will be used as validation data.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .1, random_state = random_seed)

# plt.imshow(x_train[0][:,:,0])
# plt.show()

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(learning_rate=0.001, rho=.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# To dynamically reduce the learning rate when the accuracy has stopped improving.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

# We will start using data augmentation techniques from here onwards.
# We can train our model without using data augmentation techniques as well.

# Image augmentation is a technique that is used to artificially expand the data-set.
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, 
    batch_size=batch_size), epochs=epochs, 
    validation_data = (x_val, y_val), verbose=1, 
    steps_per_epoch=x_train.shape[0]//batch_size, 
    callbacks=learning_rate_reduction)

# Evaluate the model
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history["loss"], color="b", label="Training loss")
ax[0].plot(history.history["val_loss"], color="g", label="Validation loss")
legend = ax[0].legend(loc="best", shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()


# Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap) # Each new pixel value is equal to the value of the nearest pixel in the original image.
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # It is used to compare each y value with other y values (0, 1, 2, 3, 4...).

        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Predict the values from the validation dataset
Y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) # Find the index of the maximum value
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = x_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
    plt.show()
# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)



# # predict results
# results = model.predict(test)

# # select the indix with the maximum probability
# results = np.argmax(results,axis = 1)

# results = pd.Series(results,name="Label")

