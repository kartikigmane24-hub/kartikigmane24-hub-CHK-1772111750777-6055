import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# DATASET PATH
dataset_path = dataset_path = dataset_path =
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# LOAD PRETRAINED MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# FREEZE BASE MODEL
for layer in base_model.layers:
    layer.trainable = False

# ADD NEW LAYERS
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# SAVE MODEL
model.save("plant_disease_model.h5")

print("✅ Model Training Complete!")
