from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.core.urlresolvers import reverse

from micapp.models import PicUpload
from micapp.forms import ImageForm

# Create your views here.
def index(request):
    return render(request, 'index.html')


def list(request):
     image_path = ''
     image_path1=''
     if request.method == "POST":
         form = ImageForm(request.POST, request.FILES)

         if form.is_valid():
             newdoc = PicUpload(imagefile=request.FILES['imagefile'])
             newdoc.save()

             return HttpResponseRedirect(reverse('list'))

     else:
         form = ImageForm()

     documents = PicUpload.objects.all()
     for document in documents:
         image_path = document.imagefile.name
         image_path1 = '/' + image_path

         document.delete()

     request.session['image_path'] = image_path

     return render(request, 'list.html',
     {'documents':documents, 'image_path1': image_path1, 'form':form}
     )



# #***************************** Car Damage Detection ***************************
# #******************************************************************************
# #*********************************** Start ************************************
#
# #******************************* Import essentials ****************************
import os
import json

import h5py
import numpy as np
import pickle as pk
from PIL import Image


# keras imports
from keras.models import  load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
import tensorflow as tf

#************************* Prepare Image for processing ***********************

def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Loading  valid categories for identifying cars using VGG16
with open('static/cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)

# shortlisting top 27 Categories that VGG16 stores for cars (Can be altered for less or more)
cat_list  = [k for k, v in cat_counter.most_common()[:27]]

global graph
graph = tf.get_default_graph()
#******************************************************************************

#******************************************************************************
#~~~~~~~~~~~~~~~ Prapare the flat image~~~~~~~~~~~~~
#******************************************************************************
def prepare_flat(img_224):
    base_model = load_model('static/vgg16.h5')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat

#******************* Loading Models, Weights and Categories Done **************

#******************************************************************************
#~~~~~~~~~~~~~~~~~~~~~~~~~ FIRST Check- CAR OR NOT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#******************************************************************************

CLASS_INDEX_PATH = 'static/imagenet-class-index.json'

def get_predictions(preds, top=5):

    global CLASS_INDEX
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def img_categories_check(img_224):
    first_check = load_model('static/vgg16.h5')
    print ("Validating that this is a microscope image...")
    out = first_check.predict(img_224)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in cat_list:
            print ("Image Check Passed!!!")
            print ("\n")
            return True
    return False


def image_class(img_flat):
    print ("Classifying the image...")
    fourth_check = pk.load(open("static/classifier.pickle", 'rb'))
    train_labels = ['Biological', 'Fibres','Film coated surface', 'MEMS devices and electrodes','Nanowires', 'Particles','Patterned surface', 'Porous sponge','Powder','Tips']
    preds = fourth_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print ("This is an image of - " + train_labels[preds[0]])
    print ("Image classification completed")
    print ("\n")
    print ("Thank you for using Mic!")
    return prediction


# load models
def engine(request):

    MyImg = request.session['image_path']
    img_path = MyImg
    request.session.pop('image_path', None)
    request.session.modified = True
    with graph.as_default():

        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        g1 = img_categories_check(img_224)
        g2 = image_class(img_flat)

        while True:
            try:

                if g1 is False:
                    g1_pic = "This does not look like a microscopy image!"

                    break
                else:
                    g1_pic = "This looks like a microscopy image!"

                    #g2 = image_class(img_flat)

                    break

            except:
                break


    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg") :
            os.remove(src + image_file_name)

    K.clear_session()

    context={'g1_pic':g1_pic, 'typ':g2}


    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ENGINE ENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#******************************************************************************
