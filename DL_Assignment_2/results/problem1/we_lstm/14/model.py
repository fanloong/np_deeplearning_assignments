# Build the Model 13
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Embedding(max_words, 64, input_length=maxlen))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, kernel_regularizer=l2(0.0005), return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.4))
model.add(Dense(5, activation='softmax'))
model.summary()

# Train the Model 
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=25,
                    batch_size=128,
                    validation_split=0.2)