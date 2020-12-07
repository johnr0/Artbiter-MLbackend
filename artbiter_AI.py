from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KDTree
import os
import numpy as np
import pickle
import math
from PIL import Image
import matplotlib.pyplot as plt
import re
import base64
from io import BytesIO
import json

from cav import *

image_data_path = '../../Data/wikiart/'

global vgg_layers
vgg = scipy.io.loadmat(VGG_MODEL)
vgg_layers = vgg['layers']
print("VGG matrix loaded...")

graph = tf.Graph()
build_graph(graph, vgg_layers)
print("Tensorflow graph built...")
del vgg # to free up memory

artlist = pickle.load(open('final_artlist.pkl', 'rb'))

    # the PCA model
PCAmodel = pickle.load(open('PCA_model.pkl', 'rb'))
tree = pickle.load(open('final_tree.pickle', 'rb'))

random_sampled = np.random.choice(tree.get_arrays()[0].shape[0], 20, replace=False)
random_sampled = tree.get_arrays()[0][random_sampled]


app = FlaskAPI(__name__)

background = Image.new('RGB', (224, 224), (255,255,255))

@app.route('/example/', methods=['GET', 'POST'])
def example():
    return {'hello': 'world'}

@app.route('/image_to_embedding', methods=['GET', 'POST'])
def image_to_embedding():
    if request.method == 'POST':
        # print(request.data.keys())
        image_data = re.sub('^data:image/.+;base64.', '',request.data['image'])
        # print(image_data[0:10])
        byte_data = base64.b64decode(image_data)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        img = img.resize((224, 224))
        
        if img.mode=='RGBA' or img.mode=='L':
            img = img.convert('RGB')
            # print('to rgb', img)
            # img = Image.alpha_composite(background, img)
        # print(img.mode)
        img_array = np.asarray(img)
        img_array = img_array.reshape((1, 224, 224, 3))
        # print(img_array.shape)
        with tf.Session(graph=graph) as sess:
            embeddings = images2embeddings(img_array, sess, PCAmodel)
            # print(embeddings[0])
            embedding = json.dumps(embeddings[0].tolist())
            return {'message': 'returning embedding', 'embedding': embedding}

    return {'message': 'No GET ability'}

@app.route('/trainCAV', methods=['GET', 'POST'])
def trainCAV():
    if request.method == 'POST':
        # print(request.data)
        embeddings = json.loads(request.data['embeddings'])
        for key in embeddings:
            embeddings[key] = np.asarray(embeddings[key])
            print(embeddings[key].shape)
        print(embeddings.keys())
        if len(embeddings.keys())==1:
            cavs = train_concepts(embeddings, random_sampled)
        else:
            cavs = train_concepts(embeddings)
        print(cavs)
        for key in cavs:
            cavs[key] = cavs[key].tolist()
        return {'cavs': json.dumps(cavs)}

    return {'message': 'No GET ability'}

@app.route('/searchImages', methods=['GET', 'POST'])
def searchImages():
    if request.method=='POST':
        search_embedding = np.asarray(json.loads(request.data['search_start_image_embedding']))
        cavs = json.loads(request.data['cavs'])
        search_slider_values = json.loads(request.data['search_slider_values'])

        for group_key in search_slider_values:
            search_embedding = search_embedding + search_slider_values[group_key] * np.asarray(cavs[group_key])
        # print('search embedding', search_embedding)

        searched_indexes = tree.query(search_embedding.reshape((1, 300)), k=10)
        print('searched indexes', searched_indexes[1][0])
        returned_images = []
        for idx in searched_indexes[1][0]:
            image_file = base64.b64encode(open(os.path.join(image_data_path, artlist[idx]), 'rb').read()).decode()
            image_file = 'data:image/png;base64,{}'.format(image_file)
            print(image_file)
            returned_images.append(image_file)
        
        return {'returned_images': json.dumps(returned_images)}

    return {'message': 'No GET ability'}



if __name__=="__main__":
    app.run(debug=True)