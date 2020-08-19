# Build the Model
from tensorflow.keras.layers import LSTM

# Build the Model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=maxlen)) # maxwords = 30000
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(5, activation='softmax'))
model.summary()

# Train the Model
model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=25,
                    batch_size=128,
                    validation_split=0.2)