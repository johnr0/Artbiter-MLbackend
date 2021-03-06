import sys
import os
from scipy import ndimage, misc
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import scipy.io
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = {}
vgg_layers = None
NUM_CHANNELS = [64, 128, 256, 512, 512]
LAYER_IM_SIZE = [224, 112, 56, 28, 14]
EMBED_SIZE = sum(map(lambda x:x*x, NUM_CHANNELS))
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

def build_graph(graph):
    global model

    # extract weights and biases for a given convolutional layer
    def weights_and_biases(layer_index):
        W = tf.constant(vgg_layers[0][layer_index][0][0][2][0][0])
        b = vgg_layers[0][layer_index][0][0][2][0][1]
        # need to reshape b since each element is wrapped in length 1 array
        b = tf.constant(np.reshape(b, (b.size)))
        layer_name = vgg_layers[0][layer_index][0][0][0][0]
        return W,b

    with graph.as_default():
        model['image'] = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        W,b = weights_and_biases(0)
        model['conv1_1'] = tf.nn.conv2d(model['image'], W, [1,1,1,1], 'SAME') + b
        model['relu1_1'] = tf.nn.relu(model['conv1_1'])
        W,b = weights_and_biases(2)
        model['conv1_2'] = tf.nn.conv2d(model['relu1_1'], W, [1,1,1,1], 'SAME') + b
        model['relu1_2'] = tf.nn.relu(model['conv1_2'])
        model['pool1'] = tf.nn.avg_pool(model['relu1_2'], ksize=[1, 2, 2, 1], 
            strides=[1, 2, 2, 1], padding='SAME')
        W,b = weights_and_biases(5)
        model['conv2_1'] = tf.nn.conv2d(model['pool1'], W, [1,1,1,1], 'SAME') + b
        model['relu2_1'] = tf.nn.relu(model['conv2_1'])
        W,b = weights_and_biases(7)
        model['conv2_2'] = tf.nn.conv2d(model['relu2_1'], W, [1,1,1,1], 'SAME') + b
        model['relu2_2'] = tf.nn.relu(model['conv2_2'])
        model['pool2'] = tf.nn.avg_pool(model['relu2_2'], ksize=[1,2,2,1], 
            strides=[1,2,2,1], padding='SAME')
        W,b = weights_and_biases(10)
        model['conv3_1'] = tf.nn.conv2d(model['pool2'], W, [1,1,1,1], 'SAME') + b
        model['relu3_1'] = tf.nn.relu(model['conv3_1'])
        W,b = weights_and_biases(12)
        model['conv3_2'] = tf.nn.conv2d(model['relu3_1'], W, [1,1,1,1], 'SAME') + b
        model['relu3_2'] = tf.nn.relu(model['conv3_2'])
        W,b = weights_and_biases(14)
        model['conv3_3'] = tf.nn.conv2d(model['relu3_2'], W, [1,1,1,1], 'SAME') + b
        model['relu3_3'] = tf.nn.relu(model['conv3_3'])
        W,b = weights_and_biases(16)
        model['conv3_4'] = tf.nn.conv2d(model['relu3_3'], W, [1,1,1,1], 'SAME') + b
        model['relu3_4'] = tf.nn.relu(model['conv3_4'])
        model['pool3'] = tf.nn.avg_pool(model['relu3_4'], ksize=[1,2,2,1], 
            strides=[1,2,2,1], padding='SAME')
        W,b = weights_and_biases(19)
        model['conv4_1'] = tf.nn.conv2d(model['pool3'], W, [1,1,1,1], 'SAME') + b
        model['relu4_1'] = tf.nn.relu(model['conv4_1'])
        W,b = weights_and_biases(21)
        model['conv4_2'] = tf.nn.conv2d(model['relu4_1'], W, [1,1,1,1], 'SAME') + b
        model['relu4_2'] = tf.nn.relu(model['conv4_2'])
        W,b = weights_and_biases(23)
        model['conv4_3'] = tf.nn.conv2d(model['relu4_2'], W, [1,1,1,1], 'SAME') + b
        model['relu4_3'] = tf.nn.relu(model['conv4_3'])
        W,b = weights_and_biases(25)
        model['conv4_4'] = tf.nn.conv2d(model['relu4_3'], W, [1,1,1,1], 'SAME') + b
        model['relu4_4'] = tf.nn.relu(model['conv4_4'])
        model['pool4'] = tf.nn.avg_pool(model['relu4_4'], ksize=[1,2,2,1], 
            strides=[1,2,2,1], padding='SAME')
        W,b = weights_and_biases(28)
        model['conv5_1'] = tf.nn.conv2d(model['pool4'], W, [1,1,1,1], 'SAME') + b
        model['relu5_1'] = tf.nn.relu(model['conv5_1'])
        W,b = weights_and_biases(30)
        model['conv5_2'] = tf.nn.conv2d(model['relu5_1'], W, [1,1,1,1], 'SAME') + b
        model['relu5_2'] = tf.nn.relu(model['conv5_2'])
        W,b = weights_and_biases(32)
        model['conv5_3'] = tf.nn.conv2d(model['relu5_2'], W, [1,1,1,1], 'SAME') + b
        model['relu5_3'] = tf.nn.relu(model['conv5_3'])
        W,b = weights_and_biases(34)
        model['conv5_4'] = tf.nn.conv2d(model['relu5_3'], W, [1,1,1,1], 'SAME') + b
        model['relu5_4'] = tf.nn.relu(model['conv5_4'])
        model['pool5'] = tf.nn.avg_pool(model['relu5_4'], ksize=[1,2,2,1], 
            strides=[1,2,2,1], padding='SAME')

# read in image as array of pixels (RGB) and truncate to 224 x 224
def get_imarray(filename):
    array = ndimage.imread(filename, mode='RGB')
    array = np.asarray([misc.imresize(array, (224, 224))])
    return array

def gram_matrix(F, N, M):
    # F is the output of the given convolutional layer on a particular input image
    # N is number of feature maps in the layer
    # M is the total number of entries in each filter
    Ft = np.reshape(F, (M, N))
    return np.dot(np.transpose(Ft), Ft)

def flattened_gram(imarray, session):
    grams = np.empty([EMBED_SIZE])    
    index = 0
    for i in range(5):
        grams[index:(NUM_CHANNELS[i]**2 + index)] = gram_matrix(
            session.run(model['conv' + str(i+1) + '_1'], 
                feed_dict={model['image']: imarray}), 
            NUM_CHANNELS[i], 
            LAYER_IM_SIZE[i]**2).flatten()
        index += NUM_CHANNELS[i]**2
    return grams

# distance between two style embeddings as defined in paper
def distance(fg1, fg2):
    dist = 0
    index = 0
    for i in range(5):
        square_1 = np.reshape(fg1[index:NUM_CHANNELS[i]**2 + index], 
            (NUM_CHANNELS[i], NUM_CHANNELS[i]))
        square_2 = np.reshape(fg2[index:NUM_CHANNELS[i]**2 + index], 
            (NUM_CHANNELS[i], NUM_CHANNELS[i]))
        index += NUM_CHANNELS[i]**2
        dist += (1.0 / (4 * NUM_CHANNELS[i]**2 * LAYER_IM_SIZE[i]**4)) * (
            np.linalg.matrix_power(square_1 - square_2, 2)).sum()
    return dist

# gets labels from csv file and returns list of labels matching given list of filenames
def get_labels(filenames, labels_csv):
    labels_dict = {}
    with open(labels_csv) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels_dict[row['filename']] = int(row['label'])
    labels_list = []
    for filename in filenames:
        labels_list.append(labels_dict[filename])
    return labels_list

def plot(plot_data, im_dir):
    embeddings = plot_data['embeddings']
    filenames = plot_data['filenames']
    labels = None
    if 'labels' in plot_data:
        labels = plot_data['labels']

    x = embeddings[:, 0]
    y = embeddings[:, 1]

    def onpick(event):
        nonlocal subfig,ax_list
        ax_list[1].clear()
        ind = event.ind[0] # event.ind returns list -> take first element
        img = mpimg.imread(os.path.join(im_dir, str(filenames[ind])))
        ax_list[1].imshow(img, cmap='gray')    
        plt.draw()
        print(ind, filenames[ind])

    subfig, ax_list = plt.subplots(1,2)

    pid = subfig.canvas.mpl_connect('pick_event', onpick)
    
    if labels != None:
        scatter = ax_list[0].scatter(x, y, c=labels, cmap="hot", alpha=0.5, picker=True)
        subfig.subplots_adjust(left=0.25)
        cbar_ax = subfig.add_axes([0.05, 0.15, 0.05, 0.7])
        subfig.colorbar(scatter, cax=cbar_ax)
    else:
        scatter = ax_list[0].scatter(x, y, alpha=0.5, picker=True)

    plt.show()
    subfig.canvas.mpl_disconnect(pid)

def main():
    global vgg_layers
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('im_dir', help="directory containing the input images")
    parser.add_argument('-l', '--labels', 
        help="csv file containing numerical image labels by filename")
    parser.add_argument('-d', '--dump', 
        help="location to dump the 2D image style embeddings")
    
    args = parser.parse_args()

    # load VGG model matrix from file
    vgg = scipy.io.loadmat(VGG_MODEL)
    vgg_layers = vgg['layers']
    print("VGG matrix loaded...")

    # build the tensorflow graph
    graph = tf.Graph()
    build_graph(graph)
    print("Tensorflow graph built...")
    del vgg # to free up memory

    im_dirs = [
        '../Data/wikiart/Fauvism',
        '../Data/wikiart/High_Renaissance',
        '../Data/wikiart/Impressionism',
        '../Data/wikiart/Mannerism_Late_Renaissance',
        '../Data/wikiart/Minimalism',
        '../Data/wikiart/Naive_Art_Primitivism',
        '../Data/wikiart/New_Realism',
        '../Data/wikiart/Northern_Renaissance',
        '../Data/wikiart/Pointillism',
        '../Data/wikiart/Pop_Art',
        '../Data/wikiart/Post_Impressionism',
        '../Data/wikiart/Realism',
        '../Data/wikiart/Rococo',
        '../Data/wikiart/Romanticism',
        '../Data/wikiart/Symbolism',
        '../Data/wikiart/Synthetic_Cubism',
        '../Data/wikiart/Ukiyo_e'
    ]

    for im_dir in im_dirs:
        print(im_dir)
    # get image filenames from image directory
        filenames = []
        for filename in os.listdir(im_dir):
            if os.path.splitext(filename)[1] in ('.jpg', '.png'):
                filenames.append(os.path.split(filename)[1])

        if len(filenames)>2000:
            embeddings = np.empty([2000, EMBED_SIZE])
        else:
            embeddings = np.empty([len(filenames), EMBED_SIZE])
        saved_counter = 0
        with tf.Session(graph=graph) as sess:
            count = 0
            for filename in filenames:
                embeddings[count-saved_counter, :] = flattened_gram(
                    get_imarray(os.path.join(im_dir ,filename)), sess)
                count += 1
                if count % 10 == 0:
                    print("Embedded " + str(count) + " images")
                if count % 2000 == 0 or count==len(filenames):
                    print("Storing " + str(count-saved_counter) + " images at count "+str(count))
                    # plot_data = {'embeddings': embeddings, 'filenames': filenames[saved_counter:count]}
                    if args.dump != None:
                        with open(os.path.join(args.dump, os.path.split(im_dir)[1])+'_intermediate_embeddings'+str(count)+'.npy', 'wb') as fp:
                            np.save(fp, embeddings)
                        pickle.dump( filenames[saved_counter:count], 
                            open( os.path.join(args.dump, os.path.split(im_dir)[1]) + '_embed_'+str(count)+'.pickle', "wb" ) )
                        print(embeddings.shape, len(filenames[saved_counter:count]))
                    saved_counter = count
                    if len(filenames)-count<2000:
                        del embeddings
                        embeddings = np.empty([len(filenames)-count, EMBED_SIZE])
                    else:
                        del embeddings
                        embeddings = np.empty([2000, EMBED_SIZE])


    # print("Large embeddings generated. Shape: " + str(embeddings.shape))

    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, metric=distance)
    # print("Projecting onto two dimensions... this might take a while")
    # two_d_embeddings = tsne.fit_transform(embeddings)
    # print("2D embeddings generated")

    # plot_data = {'embeddings': embeddings, 'filenames': filenames}

    # if args.labels != None:
    #     labels = get_labels(filenames, args.labels)
    #     plot_data['labels'] = labels

    # if args.dump != None:
    #     pickle.dump( plot_data, 
    #         open( os.path.join(args.dump, os.path.split(args.im_dir)[1]) + '_embed.pickle', "wb" ) )
    #     print("Pickle dumped")

    # plot(plot_data, args.im_dir)

if __name__ == "__main__":
    main()



