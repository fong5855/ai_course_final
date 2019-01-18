try:
    import datetime
    starttime = datetime.datetime.now()
    
    import matplotlib as mpl
    mpl.use('Agg')
    
    import matplotlib
    #matplotlib.use('Agg')
    import os
    import sys
    import random
    import math
    import re
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import keras
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    tf.reset_default_graph()
    # Root directory of the project
    #ROOT_DIR = os.path.abspath("../../")

    # Import Mask RCNN
    #sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    from mrcnn.config import Config
    #from mrcnn.utils import compute_overlaps_masks
    import mrcnn.model_resnet32_origin as modellib1
    #import mrcnn.model_resnet50_origin as modellib2
    #from mrcnn import visualize
    from mrcnn.model_resnet32_origin import log
    from PIL import Image
    import yaml

    #get_ipython().magic('matplotlib inline')

    # Directory to save logs and trained model
    MODEL_DIR = "/mnt/logs"

    # Local path to trained weights file
    #COCO_MODEL_PATH = "./home/zxc44052000/data/Mask_RCNN-master/mask_rcnn_coco_smoother.h5"
    # Download COCO trained weights from Releases if needed
    '''if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)'''

    iter_num = 0


    class ShapesConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 2  # background + 3 shapes

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 768 #832 #512
        IMAGE_MAX_DIM = 768 #832 #512

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8*6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 32

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 75

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 25

    config = ShapesConfig()

    #config.display()


    # ## Notebook Preferences

    # In[3]:


    def get_ax(rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

    class DrugsDataset(utils.Dataset):

        def get_obj_index(self, image):
            n = np.max(image)
            return n

        def from_yaml_get_class(self, image_id):
            info = self.image_info[image_id]
            with open(info['yaml_path']) as f:
                temp = yaml.load(f.read())
                labels = temp['label_names']
                del labels[0]
            return labels

        def draw_mask(self, num_obj, mask, image,image_id):
            #print("draw_mask-->",image_id)
            #print("self.image_info",self.image_info)
            info = self.image_info[image_id]
            #print("info-->",info)
            #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
            for index in range(num_obj):
                for i in range(info['width']):
                    for j in range(info['height']):
                        #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                        at_pixel = image.getpixel((i, j))
                        if at_pixel == index + 1:
                            mask[j, i, index] = 1
            return mask

        '''def load_shapes(self, count, height, width):
            """Generate the requested number of synthetic images.
            count: number of images to generate.
            height, width: the size of the generated images.
            """
            # Add classes
            self.add_class("shapes", 1, "square")
            self.add_class("shapes", 2, "circle")
            self.add_class("shapes", 3, "triangle")
    
            # Add images
            # Generate random specifications of images (i.e. color and
            # list of shapes sizes and locations). This is more compact than
            # actual images. Images are generated on the fly in load_image().
            for i in range(count):
                bg_color, shapes = self.random_image(height, width)
                self.add_image("shapes", image_id=i, path=None,
                               width=width, height=height,
                               bg_color=bg_color, shapes=shapes)'''

        def load_shapes(self, count, height,width,img_floder, mask_floder, imglist, dataset_root_path):
            """Generate the requested number of synthetic images.
            count: number of images to generate.
            height, width: the size of the generated images.
            """
            # Add classes
            self.add_class("shapes", 1, "butterfly")
            self.add_class("shapes", 2, "eyespot")
            #self.add_class("shapes", 3, "eyespot")
            #self.add_class("shapes", 1, "leaf")
            for i in range(count):
                filestr = imglist[i].split(".")[0]
                #print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
                #print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
                # filestr = filestr.split("_")[1]
                mask_path = mask_floder + "/" + filestr + ".png"
                yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
                #print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
                #cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")

                self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],

                            width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)

        def load_mask(self, image_id):
            """Generate instance masks for shapes of the given image ID.
            """
            global iter_num
            #print("image_id",image_id)
            info = self.image_info[image_id]
            count = 1  # number of object
            img = Image.open(info['mask_path'])
            num_obj = self.get_obj_index(img)
            mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
            mask = self.draw_mask(num_obj, mask, img,image_id)
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion


                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            labels = []
            labels = self.from_yaml_get_class(image_id)
            labels_form = []
            for i in range(len(labels)):
                if labels[i].find("butterfly") != -1:
                    # print "box"
                    labels_form.append("butterfly")
                elif labels[i].find("eyespot") != -1:
                    # print "box"
                    labels_form.append("eyespot")
            class_ids = np.array([self.class_names.index(s) for s in labels_form])
            return mask, class_ids.astype(np.int32)

        '''def load_image(self, image_id):
            """Generate an image from the specs of the given image ID.
            Typically this function loads the image from a file, but
            in this case it generates the image on the fly from the
            specs in image_info.
            """
            info = self.image_info[image_id]
            bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
            image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
            image = image * bg_color.astype(np.uint8)
            for shape, color, dims in info['shapes']:
                image = self.draw_shape(image, shape, dims, color)
            return image'''

        def image_reference(self, image_id):
            """Return the shapes data of the image."""
            info = self.image_info[image_id]
            if info["source"] == "shapes":
                return info["shapes"]
            else:
                super(self.__class__).image_reference(self, image_id)

        '''def load_mask(self, image_id):
            """Generate instance masks for shapes of the given image ID.
            """
            info = self.image_info[image_id]
            shapes = info['shapes']
            count = len(shapes)
            mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
            for i, (shape, _, dims) in enumerate(info['shapes']):
                mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                    shape, dims, 1)
            # Handle occlusions
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count-2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # Map class names to class IDs.
            class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
            return mask.astype(np.bool), class_ids.astype(np.int32)'''


        '''def draw_shape(self, image, shape, dims, color):
            """Draws a shape from the given specs."""
            # Get the center x, y and the size s
            x, y, s = dims
            if shape == 'square':
                cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
            elif shape == "circle":
                cv2.circle(image, (x, y), s, color, -1)
            elif shape == "triangle":
                points = np.array([[(x, y-s),
                                    (x-s/math.sin(math.radians(60)), y+s),
                                    (x+s/math.sin(math.radians(60)), y+s),
                                    ]], dtype=np.int32)
                cv2.fillPoly(image, points, color)
            return image'''


        def random_shape(self, height, width):

            shape = random.choice(["square", "circle", "triangle"])
            # Color
            color = tuple([random.randint(0, 255) for _ in range(3)])
            # Center x, y
            buffer = 20
            y = random.randint(buffer, height - buffer - 1)
            x = random.randint(buffer, width - buffer - 1)
            # Size
            s = random.randint(buffer, height//4)
            return shape, color, (x, y, s)

        def random_image(self, height, width):

            bg_color = np.array([random.randint(0, 255) for _ in range(3)])
            # Generate a few random shapes and record their
            # bounding boxes
            shapes = []
            boxes = []
            N = random.randint(1, 4)
            for _ in range(N):
                shape, color, dims = self.random_shape(height, width)
                shapes.append((shape, color, dims))
                x, y, s = dims
                boxes.append([y-s, x-s, y+s, x+s])
            # Apply non-max suppression wit 0.3 threshold to avoid
            # shapes covering each other
            keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
            shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
            return bg_color, shapes


    dataset_root_path="/mnt/"
    img_floder = dataset_root_path + "butterfly_pic"
    mask_floder = dataset_root_path + "butterfly_mask"
    #yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder)
    img_list_train = imglist[:75]
    img_list_test = imglist[75:100]
    count_train = len(img_list_train)
    count_test = len(img_list_test)
    # Training dataset
    dataset_train = DrugsDataset()
    dataset_train.load_shapes(count_train,768,768,img_floder, mask_floder, img_list_train,dataset_root_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DrugsDataset()
    dataset_val.load_shapes(count_test,768,768,img_floder, mask_floder, img_list_test,dataset_root_path)
    dataset_val.prepare()


    '''model1 = modellib1.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)'''
    model1 = modellib1.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    '''init_with = "last"
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        model.load_weights("/home/zxc44052000/data/Mask_RCNN-master/logs/mask_rcnn_last.h5", by_name=True)'''


    '''hist1 = model1.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=6000,
                layers="all")'''
    hist1 = model1.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers="heads")

    fig1 = plt.figure(1)
    plt.plot(hist1.history['loss'])
    #plt.plot(hist2.history['loss'])
    plt.plot(hist1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss','val_loss'], loc='best')
    fig1.savefig('/mnt/loss.png')
    '''fig2 = plt.figure(2)
    plt.plot(hist1.history['rpn_class_loss'])
    plt.plot(hist2.history['rpn_class_loss'])
    plt.title('model rpn class loss')
    plt.ylabel('rpn class loss')
    plt.xlabel('epoch')
    plt.legend(['resnet152 backbone_unet','resnet152 backbone_fcn'], loc='best')
    fig2.savefig('/home/zxc44052000/data/Mask_RCNN-master/rpn class loss.png')
    fig3 = plt.figure(3)
    plt.plot(hist1.history['rpn_bbox_loss'])
    plt.plot(hist2.history['rpn_bbox_loss'])
    plt.title('model rpn bbox loss')
    plt.ylabel('rpn bbox loss')
    plt.xlabel('epoch')
    plt.legend(['resnet152 backbone_unet','resnet152 backbone_fcn'], loc='best')
    fig3.savefig('/home/zxc44052000/data/Mask_RCNN-master/rpn bbox loss.png')
    fig4 = plt.figure(4)
    plt.plot(hist1.history['mrcnn_class_loss'])
    plt.plot(hist2.history['mrcnn_class_loss'])
    plt.title('model mrcnn class loss')
    plt.ylabel('mrcnn class loss')
    plt.xlabel('epoch')
    plt.legend(['resnet152 backbone_unet','resnet152 backbone_fcn'], loc='best')
    fig4.savefig('/home/zxc44052000/data/Mask_RCNN-master/mrcnn class loss.png')
    fig5 = plt.figure(5)
    plt.plot(hist1.history['mrcnn_bbox_loss'])
    plt.plot(hist2.history['mrcnn_bbox_loss'])
    plt.title('model mrcnn bbox loss')
    plt.ylabel('mrcnn bbox loss')
    plt.xlabel('epoch')
    plt.legend(['resnet152 backbone_unet','resnet152 backbone_fcn'], loc='best')
    fig5.savefig('/home/zxc44052000/data/Mask_RCNN-master/mrcnn bbox loss.png')
    fig6 = plt.figure(6)
    plt.plot(hist1.history['mrcnn_mask_loss'])
    plt.plot(hist2.history['mrcnn_mask_loss'])
    plt.title('model mrcnn mask loss')
    plt.ylabel('mrcnn mask loss')
    plt.xlabel('epoch')
    plt.legend(['resnet152 backbone_unet','resnet152 backbone_fcn',], loc='best')
    fig6.savefig('/home/zxc44052000/data/Mask_RCNN-master/mrcnn mask loss.png')'''
    endtime = datetime.datetime.now()
    print((endtime - starttime))
    f = open("/mnt/ok.txt",'w')
    f.write(str(starttime) + '\n' + str((endtime - starttime)))
    f.close()

except Exception as e:
    import traceback
    error_msg = traceback.format_exc()
    f = open("/mnt/error.txt",'w')
    f.write(str(error_msg))
    f.close()
        
