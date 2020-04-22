from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time

#for text cleaning
def remove_punctuation(word):
word = re.sub(r"([?.!])", r" \1 ", word)
word = re.sub(r'[" "]+', " ", word)
word = word.rstrip().strip()
word = '<go> ' + word + ' <end>'
return word

#creating dataset
def CreateDataset(path, num_examples):
lines_dataset = io.open(path, encoding='UTF-8').read().strip().split('\n')
pair_of_words = []
pair_of_words = [[remove_punctuation(word) for word in line.split('\t')] for line in lines_dataset[:num_examples]]
return zip(*pair_of_words)



en, hi = CreateDataset("DATASET.txt", None)

#converting text into tokens and generating integer encodings
def tokenize(lang):
language_tokens = tf.keras.preprocessing.text.Tokenizer(filters='')

language_tokens.fit_on_texts(lang)


TEN = language_tokens.texts_to_sequences(lang)

TEN = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')


return TEN, language_tokens

#return tensor of input and target languages
def LoadDataset(path, num_examples=None):
target_language, input_language = CreateDataset(path, num_examples)
input_tensor, input_language_tokens = tokenize(input_language)
target_tensor, target_language_tokens = tokenize(target_language)
return input_tensor, target_tensor, input_language_tokens, target_language_tokens


num_examples = 2867
input_tensor, target_tensor, input_language, target_language = LoadDataset("DATASET.txt", num_examples)

#split dataset into train and test using 80-20 rule
train_input, test_input, train_target, test_target= train_test_split(input_tensor, target_tensor, test_size=0.2)

#representation of each statement of dataset
def Conversion(lang, tensor):
for i in tensor:
if input_language!=0:
print ("%d :: %s" % (i, lang.index_word[i]))

print ("Hindi; index-word mapping")
Conversion(input_language, train_input[0])


print ("English; index-word mapping")
Conversion(target_language, train_target[0])

#preparation of dataset
size_of_buffer = len(train_input)
size_of_batch = 64
StepEpoch = len(train_input)//size_of_batch
dim_of_embedding = 256
units = 1024
vocabulary_input_size = len(input_language.word_index)+1
vocabulary_target_size = len(target_language.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target)).shuffle(size_of_buffer)
dataset = dataset.batch(size_of_batch, drop_remainder=True)


input_batch, target_batch = next(iter(dataset))


#Encoding Layer
class encoding_layer(tf.keras.Model):
def __init__(self, vocabulary_size, dim_of_embedding, encoding_units, batch_size):
super(encoding_layer, self).__init__()
self.batch_size = batch_size
self.encoding_units = encoding_units
self.embedding = tf.keras.layers.Embedding(vocabulary_size, dim_of_embedding)
self.GRU_model = tf.keras.layers.GRU(self.encoding_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
def initialize_hidden_state(self):
return tf.zeros((self.batch_size, self.encoding_units))

def call(self, e, hidden):
e = self.embedding(e)
output_state, state = self.GRU_model(e, initial_state = hidden)
return output_state, state


encoding_layer = encoding_layer(vocabulary_input_size, dim_of_embedding, units, size_of_batch)

# sample input
hidden_layer_samp = encoding_layer.initialize_hidden_state()
output_layer_samp, hidden_layer_samp = encoding_layer(input_batch, hidden_layer_samp)



#Decoding layer
class Decoder(tf.keras.Model):
def __init__(self, vocabulary_size, dim_of_embedding, decoding_units, batch_size):
super(Decoder, self).__init__()
self.batch_size = batch_size
self.decoding_units = decoding_units
self.embedding = tf.keras.layers.Embedding(vocabulary_size, dim_of_embedding)
self.gru_model = tf.keras.layers.GRU(self.decoding_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
self.fc = tf.keras.layers.Dense(vocabulary_size)
self.attention = BahdanauAttentionImplementation(self.decoding_units)


def call(self, e, hidden, EncoderOutput):
context_vector, attention_layer_weights = self.attention(hidden, EncoderOutput)
e = self.embedding(e)
e = tf.concat([tf.expand_dims(context_vector, 1), e], axis=-1)
output_state, state = self.gru_model(e)
output_state = tf.reshape(output, (-1, output_state.shape[2]))
e = self.fc(output_state)
return e, state, attention_layer_weights


decoder = Decoder(vocabulary_target_size, dim_of_embedding, units, size_of_batch)

output_decoder_samp, _, _ = decoder(tf.random.uniform((size_of_batch, 1)),
hidden_layer_samp, output_layer_samp)


#Attention layer
class BahdanauAttentionImplementation(tf.keras.layers.Layer):
def __init__(self, units):
super(BahdanauAttentionImplementation, self).__init__()
self.weight1 = tf.keras.layers.Dense(units)
self.weight2 = tf.keras.layers.Dense(units)
self.V = tf.keras.layers.Dense(1)

def call(self, q, values):
time_axis_query = tf.expand_dims(q, 1)

score = self.V(tf.nn.tanh(
self.weight1(time_axis_query) + self.weight2(values)))

attention_layer_weights = tf.nn.softmax(score, axis=1)

context_vector = attention_layer_weights * values
context_vector = tf.reduce_sum(context_vector, axis=1)

return context_vector, attention_layer_weights




attention_layer = BahdanauAttentionImplementation(10)
attention_result, attention_layer_weights = attention_layer(hidden_layer_samp, output_layer_samp)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True, reduction='none')

def ComputeLoss(real, pred):
m = tf.math.logical_not(tf.math.equal(real, 0))
loss_compute = loss_object(real, pred)
m = tf.cast(m, dtype=loss_compute.dtype)
loss_compute = loss_compute * m
return tf.reduce_mean(loss_compute)


CheckpointDirectory = './training_checkpoints'
checkpoint_prefix = os.path.join(CheckpointDirectory, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoding_layer=encoding_layer,decoder=decoder)

@tf.function
def Training(inp, tar, EncoderHidden):
loss = 0
with tf.GradientTape() as tape:
EncoderOutput, EncoderHidden = encoding_layer(inp, EncoderHidden)
DecoderHidden = EncoderHidden
DecoderInput = tf.expand_dims([target_language.word_index['<go>']] * size_of_batch, 1)
for t in range(1, targ.shape[1]):
predictions, DecoderHidden, _ = decoder(DecoderInput, DecoderHidden, EncoderOutput)
loss = loss + ComputeLoss(targ[:, t], predictions)
DecoderInput = tf.expand_dims(targ[:, t], 1)

shape = int(targ.shape[1])
batch_loss = (loss / shape)
variables = encoding_layer.trainable_variables + decoder.trainable_variables
gradients = tape.gradient(loss, variables)
optimizer.apply_gradients(zip(gradients, variables))
return batch_loss


num_epochs = 50

for epoch in range(num_epochs):
start = time.time()
EncoderHidden = encoding_layer.initialize_hidden_state()
total_loss = 0
for (batch, (inp, targ)) in enumerate(dataset.take(StepEpoch)):
batch_loss = Training(inp, targ, EncoderHidden)
total_loss = total_loss + batch_loss
if batch % 100 == 0:
print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))
if (epoch + 1) % 2 == 0:
checkpoint.save(file_prefix = checkpoint_prefix)
temp = total_loss / StepEpoch
print('Epoch {} Loss {:.4f}'.format(epoch + 1,temp))
print('Time {} sec\n'.format(time.time() - start))



#testing
def evaluate(s):
max_length_targ = 29
max_length_inp = 29
s = remove_punctuation(s)

inp = [input_language.word_index[i] for i in s.split(' ')]

inp = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_length_inp,padding='post')

inp = tf.Conversion_to_tensor(inputs)

result = ''

hidden = [tf.zeros((1, units))]
EncoderOutput, EncoderHidden = encoding_layer(inputs, hidden)

DecoderHidden = EncoderHidden
DecoderInput = tf.expand_dims([target_language.word_index['<go>']], 0)

for i in range(max_length_targ):
predictions, DecoderHidden, attention_layer_weights = decoder(DecoderInput,DecoderHidden,EncoderOutput)

PredId = tf.argmax(predictions[0]).numpy()

result = result + target_language.index_word[PredId] + ' '

if target_language.index_word[PredId] == '<end>':
return result, s

DecoderInput = tf.expand_dims([PredId], 0)

return result, s

#printing the final predictions
def translate(s):
result, s = evaluate(s)

print('Hindi Input: %s' % (s))
print('Predicted English Translation: {}'.format(result))


checkpoint.restore(tf.train.latest_checkpoint(CheckpointDirectory))

