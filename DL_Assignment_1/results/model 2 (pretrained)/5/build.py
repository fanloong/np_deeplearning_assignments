# Setting up the conv base
img_size = 150

conv_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(img_size, img_size, 3))
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'conv4_block1_1_conv':
        set_trainable = True # after black5_conv1, set_trainable becomes True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Preparing the data generators
train_datagen = ImageDataGenerator(
      rescale=1./255,
      horizontal_flip=True,
      rotation_range=10,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=20)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=20)

# Build the Model
model2 = models.Sequential()
model2.add(conv_base)
model2.add(layers.Flatten())
# model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(0.005)))
model2.add(layers.Dense(10, activation='softmax'))
model2.summary()

# Train the Model
model2.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history2 = model2.fit_generator(
      train_generator,
      steps_per_epoch=375,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=100,
      verbose=1)