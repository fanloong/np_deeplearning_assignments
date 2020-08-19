# Convert the texts and labels into numeric tensors

maxlen = 34
training_samples = len(texts)
# validation_samples = 5000 #math.floor(training_samples * 0.2)
max_words = 56702 # unique words in vocab list

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen, padding='post')

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the X & y into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state = 24)
# Refer the report Appendix
# Please enter the random_state assigned to your group