import numpy as np
from keras.utils import to_categorical, normalize
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

(train_x , train_y) , (test_x , test_y) = fashion_mnist.load_data()

print('Shape of training data: ',train_x.shape,train_y.shape)
print('shape of testing data: ',test_x.shape,test_y.shape)

train_x = normalize(train_x, axis=1)
test_x = normalize(test_x, axis=1)
classes = np.unique(train_y)
nclasses = len(classes)
print('No of unique classes: ',nclasses)
print('Output Classes: ',classes)

plt.figure(figsize=[5,5])

plt.subplot(121)
plt.imshow(test_x[1,:,:],cmap='gray')
plt.title('Ground Truth: {}'.format(test_y[1]))

plt.subplot(122)
plt.imshow(test_x[0,:,:],cmap='gray')
plt.title('Ground Truth: {}'.format(test_y[0]))
plt.figure()
plt.show()

train_x = train_x.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255
test_x = test_x / 255
print(train_x.shape , test_x.shape)
train_y_one_hot = to_categorical(train_y)
test_y_one_hot = to_categorical(test_y)

print('original label: ', train_y[0])
print('one hot: ', train_y_one_hot[0])

train_x, valid_x, train_label, valid_label = train_test_split(train_x, train_y_one_hot, test_size=0.2, random_state=13)
print(train_x.shape , valid_x.shape, train_label.shape, valid_label.shape)

batch_size = 64
epochs = 20
num_classes = 10


fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, kernel_size=(3,3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128,kernel_size=(3,3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.summary()

fashion_model.compile(loss= categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

fashion_train = fashion_model.fit(train_x, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_x,valid_label))

fashion_model.save('fashion_model_dropout_1.h5py')

test_eval = fashion_model.evaluate(test_x,test_y_one_hot,verbose=1)
print('Test Loss, Test Acccuracy',test_eval[0], test_eval[1])
#plt.clf()
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
plt.show()

predicted_classes = fashion_model.predict(test_x)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
print(predicted_classes.shape, test_y.shape)
correct = np.where(predicted_classes==test_y)[0]
print('found correct labels:', len(correct))
# for i, correct in enumerate(correct[:9]):
#     plt.subplot(3,3,i+1)
#     plt.imshow(test_x[correct].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title('predicted {}, class {}'.format(predicted_classes[correct], test_y[correct]))
#     plt.tight_layout()
