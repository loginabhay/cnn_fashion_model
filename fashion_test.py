import h5py
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.metrics import classification_report

(train_x, train_y) , (test_x, test_y) = fashion_mnist.load_data()
test_x = test_x.reshape(-1,28,28,1)


model = load_model('fashion_model_dropout_2.h5py')

predicted_classes = model.predict(test_x)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
print(predicted_classes.shape, test_y.shape)
correct = np.where(predicted_classes==test_y)[0]
print('found correct labels:', len(correct))
for i, correct in enumerate(correct[:12]):
    plt.subplot(4,3,i+1)
    plt.imshow(test_x[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('predicted {}, class {}'.format(predicted_classes[correct], test_y[correct]))
    plt.tight_layout()
plt.show()

wrong = np.where(predicted_classes != test_y)[0]
print('found wrong labels',len(wrong))
for i, wrong in enumerate(wrong[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_x[wrong].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('predicted {}, class {}'.format(predicted_classes[wrong], test_y[wrong]))
    plt.tight_layout()

plt.show()
num_classes = 10
target_names = ["Classe {}".format(i) for i in range(num_classes)]
print(classification_report(test_y, predicted_classes, target_names=target_names))