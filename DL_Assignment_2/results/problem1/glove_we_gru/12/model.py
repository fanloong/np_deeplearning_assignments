# Build the Model 12

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(GRU(64, return_sequences=True, dropout=(0.3)))
model.add(GRU(32, return_sequences=True, dropout=(0.3)))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True
model.summary()

# Train the Model
model.compile(optimizer=optimizers.RMSprop(lr=0.0017),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=25,
                    batch_size=128,
                    validation_split=0.2)