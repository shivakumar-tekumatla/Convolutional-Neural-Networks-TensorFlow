import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
from keras.callbacks import ModelCheckpoint


class CNN:
    def __init__(self) -> None:
        self.load_data() # load the data
        self.model = self.create_model()
        # Take a look at the model summary
        print(self.model.summary())
        # compile the model 
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        self.checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

        pass

    def load_data(self,n_train_val_split = 5000,normalization=True):
        # Load the fashion-mnist pre-shuffled train data and test data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # Define the text labels
        self.labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9
        if normalization:
            # Normalize the data dimenstions so that they are of approximately the same scale
            self.x_train = self.x_train.astype('float32') / 255
            self.x_test = self.x_test.astype('float32') / 255

        # Further break down the data into train/validation sets 
        (self.x_train, self.x_valid) = self.x_train[n_train_val_split:], self.x_train[:n_train_val_split] 
        (self.y_train, self.y_valid) = self.y_train[n_train_val_split:], self.y_train[:n_train_val_split]
        # Reshape input data from (28, 28) to (28, 28, 1)
        w, h = 28, 28
        self.x_train = self.x_train.reshape(self.x_train.shape[0], w, h, 1)
        self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], w, h, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], w, h, 1)

        # One-hot encode the labels
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_valid = tf.keras.utils.to_categorical(self.y_valid, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        print("Data is loaded.........!")
        # Print training set shape
        print("x_train shape:", self.x_train.shape, "y_train shape:", self.y_train.shape)

        # Print the number of training, validation, and test datasets
        print(self.x_train.shape[0], 'train set')
        print(self.x_valid.shape[0], 'validation set')
        print(self.x_test.shape[0], 'test set')
        return None
    def create_model(self):
        model = tf.keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=self.x_train.shape[1:])) 
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        print("Neural Network model is created.......!")
        
        return model

    def train(self,batch_size=64,epochs=10):
        self.model.fit(self.x_train,self.y_train,batch_size=batch_size,epochs=epochs,validation_data=(self.x_valid, self.y_valid),callbacks=[self.checkpointer])

    def test(self):
        # Load the weights with the best validation accuracy
        self.model.load_weights('model.weights.best.hdf5')   
        # Evaluate the model on test set
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score

    def visualize(self):
        y_hat = self.model.predict(self.x_test)

        # Plot a random sample of 10 test images, their predicted labels and ground truth
        figure = plt.figure(figsize=(20, 8))
        for i, index in enumerate(np.random.choice(self.x_test.shape[0], size=15, replace=False)):
            ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
            # Display each image
            ax.imshow(np.squeeze(self.x_test[index]),cmap="gray")
            predict_index = np.argmax(y_hat[index])
            true_index = np.argmax(self.y_test[index])
            # Set the title for each image
            ax.set_title("{} ({})".format(self.labels[predict_index], 
                                        self.labels[true_index]),
                                        color=("green" if predict_index == true_index else "red"))
        plt.show()
if __name__ == "__main__":
    network = CNN()
    # network.train()
    test_accuracy = network.test()
    # Print test accuracy
    print('\n', 'Test accuracy:', test_accuracy[1])
    network.visualize()

