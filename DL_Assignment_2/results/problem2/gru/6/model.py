# Build the Model
# Enter your code here:

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 2048

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = models.Sequential()
  model.add(layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]))
  model.add(layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
  model.add(layers.Dense(vocab_size))

  return model

model = build_model(len(vocab), embedding_dim, rnn_units, BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# Compile your model
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=loss)

30 epochs