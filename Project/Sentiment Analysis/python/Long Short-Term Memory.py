# import additional libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('Reddit_Data.csv')

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_comment'])
X_seq = tokenizer.texts_to_sequences(df['clean_comment'])
X_pad = pad_sequences(X_seq)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, df['category'], test_size=0.2, random_state=42)

# Build and compile the LSTM model
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=X_pad.shape[1]))
model_lstm.add(LSTM(100))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
model_lstm.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate accuracy on the test set
_, accuracy_lstm = model_lstm.evaluate(X_test, y_test)
print("Accuracy on Test Set (LSTM):", accuracy_lstm)

# Print final classification report (you may need to adjust based on your problem)
y_pred_lstm = model_lstm.predict_classes(X_test)
print("\nFinal Classification Report (LSTM):")
print(classification_report(y_test, y_pred_lstm))
