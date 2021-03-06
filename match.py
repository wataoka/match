import numpy as np
from time import sleep
from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import to_categorical

import pandas as pd
data = pd.read_csv('./result.tsv', delimiter='\t')
print(data)
quit()
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()
iteration = 1000
cnt = 0
prediction = [0.0 for i in range(10)]

acc = []
while(iteration < 100000):
    for i, image in enumerate(tqdm(x_train[0:100])):
        for j in range(iteration):
            rnd = np.random.randint(len(x_test))
            image = np.reshape(image, (28, 28))
            t_image = np.reshape(x_test[rnd], (28, 28))
            image = np.array(image)
            t_image =np.array(t_image)
            prediction[np.argmax(y_test[rnd])] += np.linalg.norm(image * t_image)
        if np.argmax(y_train[i]) == np.argmax(prediction):
            cnt += 1

        prediction = [0.0 for i in range(10)]

    print('iteration:', iteration, '\t', 'acc:', cnt / len(x_train[0:100]))
    iteration += 1000
    cnt = 0
    acc.append(cnt / len(x_train[0:100]))

print(acc)
