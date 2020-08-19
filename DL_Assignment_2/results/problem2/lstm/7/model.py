# Build the Model
# Enter your code here:
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(window_size, len(chars))))
model.add(Flatten())
model.add(layers.Dense(len(chars), activation='softmax'))
model.summary()

# Compile your model
optimizer = optimizers.Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)