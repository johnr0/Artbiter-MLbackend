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
import torch

from cav import *
from preprocessing import *

from SANet import SA_vgg, decoder, Transform, test_transform
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import torch.nn as nn

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

random_sampled_long = np.load('sampled_long_embedding.npy')
print('random sampled long', random_sampled_long.shape)

random_sampled = np.random.choice(tree.get_arrays()[0].shape[0], 20, replace=False)
random_sampled = tree.get_arrays()[0][random_sampled]


app = FlaskAPI(__name__)

background = Image.new('RGB', (224, 224), (255,255,255))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SA_decoder = decoder
SA_transform = Transform(in_planes = 512)
SA_vgg = SA_vgg

SA_decoder.eval()
SA_transform.eval()
SA_vgg.eval()

SA_decoder.load_state_dict(torch.load('SANet_decoder_iter_500000.pth'))
SA_transform.load_state_dict(torch.load('SANet_transformer_iter_500000.pth'))
SA_vgg.load_state_dict(torch.load('SANet_vgg_normalised.pth'))

norm = nn.Sequential(*list(SA_vgg.children())[:1])
enc_1 = nn.Sequential(*list(SA_vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(SA_vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(SA_vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(SA_vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(SA_vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
SA_decoder.to(device)
SA_transform.to(device)

def openImage(raw_img):
    image_data = re.sub('^data:image/.+;base64.', '',raw_img)
        # print(image_data[0:10])
    byte_data = base64.b64decode(image_data)
    image_data = BytesIO(byte_data)
    img =Image.open(image_data)
    if img.mode=='RGBA' or img.mode=='L':
        img = img.convert('RGB')
    return img

def imTOPIL(gen_result):
    grid = make_grid(gen_result, nrow=8, padding=2, pad_value=False,
                     normalize=False, range=None, scale_each=False)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


@app.route('/example/', methods=['GET', 'POST'])
def example():
    return {'hello': 'world'}

@app.route('/image_to_embedding', methods=['GET', 'POST'])
def image_to_embedding():
    if request.method == 'POST':
        # print(request.data.keys())
        ori_img = openImage(request.data['image'])
        # image_data = re.sub('^data:image/.+;base64.', '',request.data['image'])
        # # print(image_data[0:10])
        # byte_data = base64.b64decode(image_data)
        # image_data = BytesIO(byte_data)
        # ori_img = Image.open(image_data)
        img = preprocessing_image(ori_img, 224, 224, 224)
        # img = ori_img.resize((224, 224))
        
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
            embedding = json.dumps(embeddings[0].tolist())
        
    
        style_tf = test_transform()
        style = style_tf(img)
        style = style.to(device).unsqueeze(0)
        with torch.no_grad():
            Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
            Style5_1 = enc_5(Style4_1)
        style_send = {}
        style_send['relu4_1'] = Style4_1.tolist()
        style_send['relu5_1'] = Style5_1.tolist()
        print(type(style_send['relu4_1']), Style4_1.shape)
        style = json.dumps(style_send)
        return {'message': 'returning embedding', 'embedding': embedding, 'style': style}

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

        # styles = json.loads(request.data['styles'])
        # avg_styles = {}
        # for key in styles:
        #     avg_styles[key] = {}
        #     group_style = styles[key]
        #     for dim in group_style[0].keys():
        #         style_vec = np.zeros(group_style[0][dim].shape)
        #         for style in group_style:
        #             style_vec = style_vec+style[dim]
        #         avg_styles[key][dim] = (style_vec / len(group_style)).tolist()
            


        return {'cavs': json.dumps(cavs)}

    return {'message': 'No GET ability'}

@app.route('/trainStyleCAV', methods=['GET', 'POST'])
def trainStyleCAV():
    if request.method == 'POST':
        styles = json.loads(request.data['styles'])
        print(styles.keys())
        dims = {}
        embeddings = {}
        for group_key in styles:
            embeddings[group_key]=None
            group = styles[group_key]
            for art in group:
                print(art.keys())
                style_flatten=None
                for layer_key in art:
                    print(layer_key)
                    cur_arr = np.asarray(art[layer_key])
                    if layer_key not in dims:
                        dims[layer_key] = cur_arr.shape
                    cur_arr = cur_arr.flatten()
                    cur_arr = np.reshape(cur_arr, (1,cur_arr.shape[0]))
                    if style_flatten is None:
                        style_flatten = cur_arr
                    else:
                        style_flatten = np.concatenate((style_flatten, cur_arr), axis=1)
                        # print(style_flatten.shape)
                if embeddings[group_key] is None:
                    embeddings[group_key] = style_flatten
                else: 
                    embeddings[group_key] = np.concatenate((embeddings[group_key], style_flatten), axis=0)
            print(embeddings[group_key].shape)
        print(embeddings.keys())
        if len(embeddings.keys())==1:
            cavs = train_concepts(embeddings, random_sampled_long, dim=501760)
        else:
            cavs = train_concepts(embeddings, dim=501760)
        print(cavs.keys())
        print(dims.keys())
        fdim = dims[list(dims.keys())[0]]
        dim_len = 1
        for d in fdim:
            dim_len = dim_len*d
        print(dim_len)
        styles = {}
        for group_key in cavs:
            styles[group_key] = {}
            layers = np.split(cavs[group_key], [dim_len])
            for idx, d in enumerate(list(dims.keys())):
                print(idx, d)
                styles[group_key][d] = np.reshape(layers[idx], dims[d]).tolist()

        return  {'styles': json.dumps(styles)}

        


    return {'message': 'No GET ability'}

@app.route('/sliderImpact', methods=['GET', 'POST'])
def sliderImpact():
    if request.method=='POST':
        num = 1
        cavs = json.loads(request.data['cavs'])
        for key in cavs:
            cavs[key] = np.asarray(cavs[key])
            cavs[key] = cavs[key].reshape((1,300))
        search_slider_values = json.loads(request.data['search_slider_values'])
        print(search_slider_values)
        cur_image = np.asarray(json.loads(request.data['cur_image'])).reshape((1,300))
        # print('til here')
        distances = {}
        tree_arr = tree.get_arrays()
        # print('tree shape',tree_arr[0].shape)
        base_arr = cur_image
        for key2 in search_slider_values:
            base_arr = base_arr + search_slider_values[key2] * cavs[key2]
        for key1 in search_slider_values:
            distance = []
            vectors = []
            # print(cavs[key1].shape)
            base_arr = base_arr - search_slider_values[key1] * cavs[key1]
            
            for i in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
                cur_arr = base_arr + i * cavs[key1]
                # print(cur_arr)
                searched = tree.query(cur_arr, k=num)
                v_arr = np.zeros((1, 300))
                # print(searched[1][0])
                for idx in searched[1][0]:
                    v_arr = v_arr + tree_arr[0][idx]
                # print(v_arr/10)
                vectors.append(v_arr/num)
            standard_arr = base_arr + search_slider_values[key1] * cavs[key1]
            for j in range(9):
                # print(vectors[j]-vectors[0])
                distance.append(np.linalg.norm(vectors[j]-cur_image))
            # print(distance)
            if np.max(distance)!=0:
                distance = distance/np.max(np.abs(distance))
            # for idx, d in enumerate(distance):
            #     if np.max(distance)!=0:
            #         distance[idx] = d/np.max(np.abs(distance))
            print(distance)
            distances[key1] = list(distance)
                # print('tree', searched)
        return {'distances': json.dumps(distances)}

    return {'message': ' No GET ability'}

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

@app.route('/generateImage', methods=['GET', 'POST'])
def generateImage():
    if request.method=='POST':
        content = json.loads(request.data['content'])
        content_weight = float(request.data['content_weight'])
        for k in content:
            content[k] = torch.FloatTensor(content[k])
        styles = json.loads(request.data['styles'])
        style_weights=json.loads(request.data['style_weights'])
        style={}
        for k in content:
            s=content[k]*content_weight
            for sk in styles:
                s = s + style_weights[sk] * torch.FloatTensor(styles[sk][k])
            style[k] = s
        for k in content:
            print(k)
            print(content[k].shape, style[k].shape)
        with torch.no_grad():
            start_time = time.time()
            gen_result = SA_decoder(SA_transform(content['relu4_1'], style['relu4_1'], content['relu5_1'], style['relu5_1']))
            print('time', time.time()-start_time)
            
            gen_result.clamp(0, 255)
        gen_result = gen_result.cpu()
        print(gen_result)
        # im_np = gen_result.numpy()
        # print(im_np.shape)
        # im = Image.fromarray(np.uint8(im_np*255))
        # im = transforms.ToPILImage(mode='RGB')(gen_result[0])#.convert("RGB")
        im = imTOPIL(gen_result)
        im.save('./other_gen_test.png')
        buffer = BytesIO()
        im.save(buffer, format="JPEG")
        image_file = base64.b64encode(buffer.getvalue()).decode()
        image_file = 'data:image/png;base64,{}'.format(image_file)

        # save_image(gen_result, 'gen_test.jpg')
        return {'returned_images': json.dumps([image_file])}
        

    return {'message': 'No GET ability'}

@app.route('/generateImageWithScaling', methods=['GET', 'POST'])
def generateImageWithScaling():
    if request.method=='POST':
        content = json.loads(request.data['content'])
        styles = json.loads(request.data['styles'])

        # get the dimensions first
        print(content.keys())
        content['image']= openImage(content['content_image'])
        content['content_mask'] = openImage(content['content_mask'])
        content['content_mask'] = content['content_mask'].crop((content['content_position']['left'], content['content_position']['top'], content['content_position']['right'], content['content_position']['bottom']))
        print(content['image'].size)
        if content['image'].size[0]>content['image'].size[1]:
            content_width = 512
            content_height = int(512 * content['image'].size[1]/content['image'].size[0])
            content_side = content_height
        else:
            content_height = 512
            content_width = int(512 * content['image'].size[0]/content['image'].size[1])
            content_side = content_width
        print(content_height, content_width, content_side)
        sides = []
        sides.append(content_side)
        content['image'] = content['image'].resize([content_width, content_height])
        
        for style in styles:
            style['image'] = openImage(style['file'])
            side = np.min(style['image'].size)/np.max(style['image'].size)*style['scale']/content['content_scale']*content_side
            if style['image'].size[0]<style['image'].size[1]:
                width = int(side)
                height = int(side * style['image'].size[1]/style['image'].size[0])
                sides.append(width)
            else: 
                height = int(side)
                width = int(side * style['image'].size[0]/style['image'].size[1])
                sides.append(height)
            style['image'] = style['image'].resize((width, height))
            print(style['image'].size)
            # if()
            print(height, width)
            # sides.append(int(np.min(style['image'].size)/np.max(style['image'].size)*style['scale']/content_side * 512))
            print(style.keys())
        style_side = np.min(sides)
        print('style side:', style_side)

        
        

        with torch.no_grad():
        # center crop content, to stylle

            content['style_image'] = central_crop([content['image']], style_side, style_side)[0]
            print(content['style_image'].size)
            content['content_embedding'] = image2embedding_style(content['image'])
            content['style_embedding'] = image2embedding_style(content['style_image'])


        # crop styles
            for style in styles:
                style['image'] = central_crop([style['image']], style_side, style_side)[0]
                style['style_embedding'] = image2embedding_style(style['image'])
            
            # do transfer

            style_to_transfer = content['style_embedding']
            weight = content['content_weight']
            for k in style_to_transfer:
                style_to_transfer[k] = weight * style_to_transfer[k]
            for style in styles:
                weight = weight + style['weight']
                # print('style weight', style['weight'])
                for k in style_to_transfer:
                    style_to_transfer[k] = style_to_transfer[k]+style['style_embedding'][k]*style['weight']
                    
            for k in style_to_transfer:
                print('shape', style_to_transfer[k].size())
                print('content shape', content['content_embedding'][k].size())
                style_to_transfer[k] = style_to_transfer[k]/weight
            # print('weight',weight)
            
            gen_result = SA_decoder(SA_transform(content['content_embedding']['relu4_1'], style_to_transfer['relu4_1'], content['content_embedding']['relu5_1'], style_to_transfer['relu5_1']))
            gen_result.clamp(0, 255)
        gen_result = gen_result.cpu()
        # print(gen_result)
        im = imTOPIL(gen_result)
        # im = transforms.ToPILImage()(gen_result[0])
        # im = Image.fromarray(np.transpose(gen_result.numpy()[0],(1, 2, 0)), 'RGB')
        # im.save('./test_int0.png')
        im = im.resize((content['content_mask'].size))
        transparent_background = Image.new('RGBA', (im.size), (255,255,255,0))
        content['content_mask'] = content['content_mask'].convert('L')
        # im.save('./test_int.png')
        # print(im.size, transparent_background.size, content['content_mask'].size)
        im = Image.composite(im, transparent_background, content['content_mask'])
        im_in_image = Image.new('RGBA', (1000, 1000), (255,255,255,0))
        im_in_image.paste(im, (content['content_position']['left'], content['content_position']['top']))
        # im_in_image.save('./test.png')
        buffer = BytesIO()
        # transparent_background.save(buffer, format="PNG")
        im_in_image.save(buffer, format="PNG")
        image_file = base64.b64encode(buffer.getvalue()).decode()
        image_file = 'data:image/png;base64,{}'.format(image_file)
        # im.save('./test.png')
        
        # save_image(gen_result, './test_ori.png')
        return {'returned_image': image_file}

    return {'message': 'No GET ability'}

def image2embedding_style(image):
    style_tf = test_transform()
    style = style_tf(image)
    style = style.to(device).unsqueeze(0)
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    return {
        'relu4_1': Style4_1,
        'relu5_1': Style5_1
    }


@app.route('/randomSearchImage', methods=['GET', 'POST'])
def randomSearchImage():
    if request.method=='POST':
        # list(range(tree.get_arrays()[0].shape[0]))
        sampled = np.random.choice(tree.get_arrays()[0].shape[0], 10, replace=False)
        returned_images = []
        for idx in sampled:
            image_file = base64.b64encode(open(os.path.join(image_data_path, artlist[idx]), 'rb').read()).decode()
            image_file = 'data:image/png;base64,{}'.format(image_file)
            print(image_file)
            returned_images.append(image_file)
        
        return {'returned_images': json.dumps(returned_images)}
    return {'message': 'No GET ability'}

if __name__=="__main__":
    app.run(debug=True)