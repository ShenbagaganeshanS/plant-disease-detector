import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical'
)

val_data = test_datagen.flow_from_directory(
    'dataset/valid',
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical'
)


base_model = tf.keras.applications.MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)

base_model.trainable = False


x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


print("Initial Training Started...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=2
)



base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=2
)


model.save("model_finetuned")


loss, accuracy = model.evaluate(test_data)
print(" Final Test Accuracy:", accuracy)