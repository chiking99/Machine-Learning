from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class LanguageModel:
    def __init__(self, ngram_size, tokenizer,Dimension):
        self.ngram_size = ngram_size
        self.tokenizer = tokenizer
        self.Dimension = Dimension
        self.model = Sequential()
        # TODO: Implement
        #https://stackoverflow.com/questions/53525994/how-to-find-num-words-or-vocabulary-size-of-keras-tokenizer-when-one-is-not-as
        vocab_size=len(self.tokenizer.word_index)+1 #vocabulary size
        #Build the model
        self.model.add(layers.Embedding(input_dim=vocab_size,output_dim=Dimension,input_length=self.ngram_size-1)) 
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(10,activation='tanh'))
        self.model.add(layers.Dense(10,activation='relu'))
        self.model.add(layers.Dense(vocab_size,activation='softmax'))
        self.model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    def train(self, train_X, epochs=10, batch_size=8):
        # TODO: Implement
        #extract first 2 column
        X=train_X[:,0:(self.ngram_size-1)]
        #convert vectors to 1-of-K representation
        one_hot_train_y=np_utils.to_categorical(train_X[:,self.ngram_size-1])
        #fit model
        history = self.model.fit(X,one_hot_train_y,epochs=epochs,batch_size=batch_size)
        return history

    def predict(self, context):
        # TODO: Implement
        #add a special PAD id "0" either at the end (default) or the beginning of the example
        context_pad=pad_sequences(context,padding='post',maxlen=self.ngram_size-1)
        #predit the words
        logits= self.model.predict(context_pad)
        #compute argmax Probability
        pred_index=np.argmax(logits)
        return pred_index, logits

    def generate(self, context, max_num_words=20):
        output = []
        # TODO: Implement
        index=context[0][:]
        i=0
        while len(output) < max_num_words:
          context_2=[[index[i],index[i+1]]]
          pred_index, logits=self.predict(context_2)
          index=[index[i+1],int(pred_index)]

          word=self.tokenizer.index_word[pred_index]

          output.append(word)
        return output

    def sent_log_likelihood(self, ngrams):
        logprob = 1
        # TODO: Implement
        #predict probability and multiply it together
        for i in range(len(ngrams)):
          X=[[ngrams[i][0],ngrams[i][1]]]
          pred_index, logits= self.predict(X)
          y=ngrams[i][2]
          logprob *=logits[0][y]
        return logprob

    def fill_in(self, prefix, suffix, get_ngrams_fn):
        # TODO: Implement (MSc Students only)
        logits = []
        pred_word_id = 0
        return pred_word_id, logits

    def get_word_embedding(self, word):
        return self.model.layers[0].get_weights()[0][self.tokenizer.word_index[word]]