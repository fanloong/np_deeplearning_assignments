from tensorflow.keras.layers import LSTM

# Build the Model
model2 = Sequential()
model2.add(Embedding(max_words, 128, input_length=maxlen))
model2.add(LSTM(32, return_sequences=True))
model2.add(LSTM(32))
model2.add(Dense(5, activation='softmax'))
model2.summary()

# Train the Model
model2.compile(optimizer=optimizers.RMSprop(lr=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history2 = model2.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)