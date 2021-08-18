# UTILS


def get_tokenizer(data):
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(data['review'])

  return tokenizer


