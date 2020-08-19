# Build the Model
# Enter your code here:
model = models.Sequential()
model.add(layers.LSTM(128, return_sequences=True, input_shape=(window_size, len(chars))))
model.add(layers.LSTM(128, kernel_regularizer=l2(0.01),  return_sequences=True))
model.add(layers.LSTM(32, kernel_regularizer=l2(0.005), dropout=(0.5)))
model.add(layers.Dense(len(chars), activation='softmax'))
model.summary()

# Train the Model
# Enter your code here:
optimizer = optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

for epoch in range(1, 5):
    print('epoch', epoch)
    # Fit the model for 1 epoch on the available training data
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    # Select a text seed at random
    start_index = random.randint(0, len(text) - window_size - 1)
    generated_text = text[start_index: start_index + window_size]
    # generated_text = input("Enter text to start predicting from: ")
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, window_size, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, chars_to_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()