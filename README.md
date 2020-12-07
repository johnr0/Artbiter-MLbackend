# Before using...

Add [final_artlist.pkl](https://drive.google.com/file/d/12DfPPNz-WO3cmcj1f3g1BT4otxufnLOc/view?usp=sharing), [final_embedding.npy](https://drive.google.com/file/d/1P9XPnoifbuprkcCMSQLbujbrahxUiUnp/view?usp=sharing), [final_tree.pickle](https://drive.google.com/file/d/1yLRwcv2Yur6mwXg7HdtX6Z_JN_qr2Qbe/view?usp=sharing), and [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/pretrained/) in the directory. Also, put [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) image data somewhere and connect to the web server.

# Webserver
ML web server can be run by 
```
python artbiter_AI.py
```

# Art embedding by style

Visualize a collection of artwork images by style similarity, as defined in [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf). Specify a folder of images, and the program generates a two-dimensional scatterplot in which each data point represents an image in the folder, and images that are stylistically similar appear close together. Click on the data points to see the images they represent.

Embeddings are generated by feeding each image through the 19-layer VGG convolutional neural network, embedding them in a high-dimensional vector space as the concatenation of their flattened Gram matrices, defining the distance between two vectors to be the "style loss" between them, and finally projecting onto two dimensions with TSNE.

## Prerequisites
- All package dependencies are specified in `environment.yml`. Users of the conda package management system can copy my exact environment with the command
```
conda env create -f environment.yml
```
- Additionally, you must download the file `imagenet-vgg-verydeep-19.mat` into the project directory, found [here](http://www.vlfeat.org/matconvnet/pretrained/)

## Usage

Execute the program with
```
python embed_by_style.py [path_to_image_directory]
```
All files in the specified directory with a `.png` or `.jpg` extension will be included in the visualization.

### Options
The optional flag
```
-l [label_csv_file]
```
takes a csv file of numerical image labels. The csv file must have a heading of the form "filename,label" and must contain an entry for every image file present in the image directory. If specified, the labels will be used to color the data points in the scatter plot. Possible labels include:
- year in which the artwork was created
- numerical identifier of the artist (e.g. 0 for Picasso, 1 for Van Gogh, etc)
- artistic school


The optional flag
```
-d [path_to_dump_location]
```
will dump a pickled version of the two-dimensional embedding data into the specified location with the name `[image_folder_name]_embed.pickle`. The pickle file contains a dictionary with keys 'embeddings', 'filenames', and 'labels' (if specified).

### Help
List command line options with
```
python embed_by_style.py --help
```

## Example
A sample image folder containing selected works of Picasso (courtesty of [WikiArt](http://www.wikiart.org)) can be found in `sample/picasso`. The file `picasso.csv` labels each image by year of creation.
Running the command
```
python embed_by_style.py sample/picasso -d sample -l sample/picasso.csv
```
uses the labels in the csv file to color images on the scatter plot and dumps a pickle file `picasso_embed.pickle` in the folder `sample`. 
The following scatterplot is generated:
<img src="https://github.com/aheyman11/art_embeddings/blob/master/sample/screenshot.png"/>
