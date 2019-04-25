import pickle
import string
import numpy as np
import gensim
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import euclidean_distances
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant


def perform_tokenize(movie_data, filename):
    data_sentences = list()
    lines = movie_data['review'].values.tolist()

    for line in lines:
        tokens = word_tokenize(line)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        data_sentences.append(words)
    if filename:
        with open(filename, 'w') as myfile:
            json.dump(data_sentences, myfile)
    return data_sentences


def train_word2vec(data_sentences, embedding_dim, window_size, n_of_workers, count):
    model = gensim.models.Word2Vec(sentences=data_sentences, size=embedding_dim, window=window_size, workers= n_of_workers, min_count=count)
    # vocab size
    words = list(model.wv.vocab)
    print("vocabulary size {}".format(len(words)))
    return model


def save_the_model(model, dim, window_size):
    filename = "fileee.txt"
    model.wv.save_word2vec_format(filename, binary=False)
    return filename


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_precomputed_model(filename, embeddings_index):
    f = open(filename, encoding="utf-8")
    for i, line in enumerate(f):
        if i == 0:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    print(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.9, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_the_embeds(model):

    keys = ['like', 'photo', 'heart', 'red', 'job', 'coffee', 'happy', 'sad', 'twitter', 'haha', 'ok', '4']
    print(keys)

    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    embedding_clusters = np.array(embedding_clusters)

    print(len(word_clusters))
    print(len(embedding_clusters))

    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=5, n_components=2, init='pca', n_iter=3200, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    tsne_plot_similar_words('similar words in imdb data set', keys, embeddings_en_2d, word_clusters, 1, 'similar_words.png')


def embedding_to_matrix(movie_data, data_sentences, embedding_index, embedding_dim, max_length):
    valid_split = 0.2

    # vectorize the text samples into a 2D integer tensor
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(data_sentences)
    sequences = tokenizer_obj.texts_to_sequences(data_sentences)

    # pad sequences
    word_index = tokenizer_obj.word_index
    print('Found %s unique tokens.' % len(word_index))

    review_pad = pad_sequences(sequences, maxlen=max_length)
    sentiment = movie_data['sentiment'].values
    print('Shape of review tensor:', review_pad.shape)
    print('Shape of sentiment tensor:', sentiment.shape)

    # split the data into a training set and a validation set
    indices = np.arange(review_pad.shape[0])
    np.random.shuffle(indices)
    review_pad = review_pad[indices]
    sentiment = sentiment[indices]
    num_validation_samples = int(valid_split * review_pad.shape[0])

    x_train_pad = review_pad[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    x_test_pad = review_pad[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]

    print('Shape of x_train_pad tensor:', x_train_pad.shape)
    print('Shape of y_train tensor:', y_train.shape)

    print('Shape of x_test_pad tensor:', x_test_pad.shape)
    print('Shape of y_test tensor:', y_test.shape)

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return x_train_pad, y_train, x_test_pad, y_test, embedding_matrix, num_words

from keras.utils import plot_model

def create_train_nn(x_train, y_train, x_test, y_test, num_words, embeddings_dim, embedding_matrix, max_length):
    # define model
    model = Sequential()
    embedding_layer = Embedding(num_words,
                                embeddings_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)
    model.add(embedding_layer)
    model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print('Summary of the built model...')
    print(model.summary())


    plot_model(model, to_file="my_model.png", show_shapes=True)

    exit()
    print('Train...')
    
    history = model.fit(x_train, y_train, batch_size=256, epochs=25, validation_data=(x_test, y_test), verbose=1)
    print('Testing...')
    score, acc = model.evaluate(x_test, y_test, batch_size=256)
    
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    print("Accuracy: {0:.2%}".format(acc))

    accuracy_loss_plots(history)


def accuracy_loss_plots(history):
    # accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('imdb_accuracy.pdf')
    plt.close()
    # loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('imdb_loss.pdf')


def plot_confusion_matrix(cm, target_names):
    # when we pass a confusion matrix and the target names it produces a plot of the confusion matrix

    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(df_cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig("confusion_matrix_plot.png")


def calculate_distance(embeddings_index, max_words):
    mat = euclidean_distances(list(embeddings_index.values())[:max_words])
    # Replace self distances from 0 to inf (to use argmin)
    np.fill_diagonal(mat, np.inf)
    return mat


def save_to_file(embeddings_index, simil, embeddings_dim, max_words):
    # Save the pairs to a file
    f_out = open('similarity_pairs_dim'+str(embeddings_dim)+'_first'+str(max_words)+'.txt','w')
    for i, item in enumerate(list(embeddings_index.keys())[:max_words]):
        f_out.write(str(item)+' '+str(list(embeddings_index.keys())[simil[i]])+'\n')


def tester(embeddings_index):
    # Test the "king - man + woman = queen" analogy
    # Compute embedding of the analogy
    embedding_analogy = embeddings_index['king'] - embeddings_index['man'] + embeddings_index['woman']
    # Find distances with the rest of the words
    analogy_distances = np.empty(len(embeddings_index))
    for i, item in enumerate(list(embeddings_index.values())):
        analogy_distances[i] = euclidean_distances(embedding_analogy.reshape(1, -1), item.reshape(1, -1))
    # Print top 10 results
    a = [list(embeddings_index.keys())[i] for i in analogy_distances.argsort()[:10]]
    print(a)
    return a


def main():

    print("in main")
    # read the prerocessed data
    movie_data = pd.read_csv('movie_data.csv', encoding='utf-8')
    movie_data.head(3)

    # tokenize and get sentences
    # data_sentences = perform_tokenize(movie_data, 'sentences.txt')

    with open('sentences.txt', 'r') as infile:
        data_sentences = json.load(infile)

    # train word2vec model
    #embeddings_dim = 100
    max_words = 200
    window = 5
    workers = 4
    min_count = 10
    filename = ""

    panda_data = list()


    #pairwise eucledian manhattan
    for embedding_dim in [32, 64, 100]:
        for window in [2, 5, 10]:
            for min_count in [1, 5, 10]:
                model = train_word2vec(data_sentences, embedding_dim, window, workers, min_count)
                filename = save_the_model(model, embedding_dim, window)
                similar_words = model.wv.most_similar('horrible')
                print(similar_words)
                embeddings_index = {}
                # load precomputed word embeddings into dictionary
                embeddings_index = load_precomputed_model(filename, embeddings_index)
                dist_mat = calculate_distance(embeddings_index, max_words)
                np.fill_diagonal(dist_mat, np.inf)
                simil = np.argmin(dist_mat, axis=0)
                queen_words = tester(embeddings_index)
                panda_data.append([embedding_dim, window, min_count, queen_words, similar_words])
    df = pd.DataFrame(panda_data, columns=['Embedding dimension', 'window size', 'min count', 'king-man+women', 'similar to harrible'])
    df.to_excel("embed_tests_results.xlsx")
    exit()



    model = train_word2vec(data_sentences, embeddings_dim, window, workers, min_count)
    filename = save_the_model(model, embeddings_dim, window)

    #save_obj(model, "model_embed")
    #model = load_obj("model_embed")

    model.wv.similar_by_word("cat")
    model.wv.most_similar('horrible')

    if not filename:
        filename = "imdb_word2vec_embeddings_Dim128_window5.txt"

    # Create a dictionary/map to store the word embeddings
    embeddings_index = {}
    # load precomputed word embeddings into dictionary
    embeddings_index = load_precomputed_model(filename, embeddings_index)

    # plotting
    # plot_the_embeds(model)

     # compute the distance
    # Compute the most similar word for every word
    # Replace self distances from 0 to inf (to use argmin)
    dist_mat = calculate_distance(embeddings_index, max_words)
    np.fill_diagonal(dist_mat, np.inf)
    simil = np.argmin(dist_mat, axis=0)
    tester(embeddings_index)
    # save to file
    # save_to_file(embeddings_index, simil, embedding_dim, max_words)
    # test as per given code


    exit()

    x_train = movie_data.loc[:24999, 'review'].values
    x_test = movie_data.loc[25000:, 'review'].values

    total_reviews = x_train + x_test
    max_length = max([len(s.split()) for s in total_reviews])

    x_train, y_train, x_test, y_test, embedding_matrix, num_words = embedding_to_matrix(movie_data, data_sentences, embeddings_index, embeddings_dim, max_length)

    # create and train the network
    create_train_nn(x_train, y_train, x_test, y_test, num_words, embeddings_dim, embedding_matrix, max_length)


if __name__ == '__main__':
    main()
