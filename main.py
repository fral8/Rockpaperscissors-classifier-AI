import tensorflow as tf
import matplotlib.pyplot as plt

def create_datagen(training_dir, val_dir):
    training_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
    val_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    train_gen=training_datagen.flow_from_directory(training_dir,batch_size=32,class_mode='categorical',target_size=(150,150))
    val_gen=val_datagen.flow_from_directory(val_dir,target_size=(150,150),class_mode='categorical',batch_size=32)
    return train_gen,val_gen

def create_model():
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3),  activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(3,activation=tf.nn.softmax)

    ])
    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])
    return model

class customCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy')>0.92:
            print(f'acc over 92% so end')

            self.model.stop_training=True

def print_acc_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()

def save_model(model,path):
    model.save(path)


if __name__=='__main__':
    training_dir='C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\Rockpaperscissors\Data\\training'
    validation_dir='C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\Rockpaperscissors\Data\\validation'
    train_gen,val_gen=create_datagen(training_dir,validation_dir)
    model=create_model()
    callback=customCallback()
    history=model.fit(train_gen,validation_data=val_gen,epochs=50,callbacks=[callback])
    print_acc_loss(history)
    save_model(model,'C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\Rockpaperscissors\saved_model')


