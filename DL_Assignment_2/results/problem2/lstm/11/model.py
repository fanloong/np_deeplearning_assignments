# Build the Model
# Enter your code here:
model = models.Sequential()
model.add(layers.LSTM(512, return_sequences=True, input_shape=(window_size, len(chars))))
model.add(layers.LSTM(256, return_sequences=True, dropout=(0.2)))
model.add(layers.LSTM(128, dropout=(0.2)))
model.add(layers.Dense(len(chars), activation='softmax'))
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.0005))