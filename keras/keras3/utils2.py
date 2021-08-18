
def get_tokenizer2(data):
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(data)

  return tokenizer