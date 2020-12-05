from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os

os.chdir(r"C:\Users\LTM0110User\Desktop\vs_code")
Dataset= "./cv00-master/Datasets\dogs_cats_sample_1000"
Batch_size=16

datagen=ImageDataGenerator(validation_split= 0.2, rescale= 1/255)

train_generator = datagen.flow_from_directory( Dataset, 
    target_size= (64,64),
    class_mode= "binary",
    subset="training")

validation_generator = datagen.flow_from_directory( Dataset, 
    target_size= (64,64),
    class_mode= "binary",
    subset="validation")

print(train_generator.class_indices)

model= Sequential()
model.add(Conv2D(filters=64, kernel_size=4, padding="same",activation="relu", input_shape=(64,64,3) ))
model.add(MaxPooling2D(pool_size=3, strides=3))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=3, strides=3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation= 'relu') )
model.add(Dropout(0.5))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(optimizer="adam", loss="binary_crossentropy" , metrics=["accuracy"])

model.fit(train_generator,epochs=50)


metrics_train=model.evaluate(train_generator)
metrics_val=model.evaluate(validation_generator)
print("train accuracy %4f and loss %4f" % metrics_train[1], metrics_train[0])
print("test accuracy %4f and loss %4f" % metrics_val[1], metrics_val[0])

model.save("model_gen.h5")