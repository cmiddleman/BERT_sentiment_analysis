
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # A dependency of the preprocessing model
import numpy as np
import pickle

tf.get_logger().setLevel('ERROR')

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

sentiment_dict = {0: 'negative', 1:'positive'}

def build_classifier_model(num_classes):

    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x

    model = Classifier(num_classes)
    return model

def load_weights(pkl_path):
    """
    Loads weights for the BERT and Dense layers of the classifier model.

    Assumes the weight files are pickled and saved at 'pkl.path/bert_layer_weights.pkl'
    and 'pkl.path/dense_layer_weights.pkl' respectively.
    """

    with open(pkl_path + 'bert_layer_weights.pkl', 'rb') as f:
        bert_weights = pickle.load(f)

    with open(pkl_path + '/dense_layer_weights.pkl', 'rb') as f:
        dense_weights = pickle.load(f)

    return bert_weights, dense_weights

class BertAmazonSentiment:

    def __init__(self, pkl_path, num_classes=2):
        
        #load the preprocessing unit from tensorflow hub
        self.bert_preprocessor = hub.load(tfhub_handle_preprocess)

        #create the bert sentiment classifier
        self.bert_model = build_classifier_model(num_classes)

        #load and set the weights
        bert_weights, dense_weights = load_weights(pkl_path)
        self.bert_model.layers[0].set_weights(bert_weights)

        #make model realize what shape its dense layer is
        #unclear why this is not neccesary for loading the BERT layer weights
        _ = self.classify([''])
        self.bert_model.layers[-1].set_weights(dense_weights)

    def preprocess(self, input_strings):
        """string go in, preprocessed tokens for BERT come out.
        """
        tok = self.bert_preprocessor.tokenize(tf.constant(input_strings))
        return self.bert_preprocessor.bert_pack_inputs([tok], tf.constant(128))

    def classify(self, input_strings):
        tok = self.bert_preprocessor.tokenize(tf.constant(input_strings))
        text_preprocessed = self.bert_preprocessor.bert_pack_inputs([tok], tf.constant(128))

        logits = self.bert_model(text_preprocessed)
        return tf.keras.activations.softmax(logits)

    def pretty_classify_one(self, review_text):
        """classify one review text and print the result as a user friendly string.
        """
        bert_result = self.classify([review_text])[0]
        idx = np.argmax(bert_result)

        return 'BERT is {0:.1f}% sure your review is {1}.'.format(100*bert_result[idx], sentiment_dict[idx])
