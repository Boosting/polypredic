import os
import numpy as np
import wget
import argparse
import base64
import json
import time
import torch
from Utils import utils
import cityscapes
import polyrnnpp
import sys
import cv2
#import polyinteractor
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import skimage.io as io

from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global instance#, fig, ax
instance = {}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', required=True)
    parser.add_argument('--reloadmodel', required=True)
    parser.add_argument('--image_dir', default='static/images')
    parser.add_argument('--port', default=5001)

    args = parser.parse_args()
    return args

def get_data_loaders(opts, DataProvider):
    print'Building dataloaders'
    data_loader = DataProvider(split='val', opts=opts['train_val'], mode='tool')

    return data_loader


class PolygonInteractor(object):
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

      'r' will do the fixing part over here. Will pass the new coordinates to the model for the second round of the prediction

    """
    showverts = True
    global corr_cords
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):

        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)		## collects all the polygon x,y coordinates and then connects the same using the Line2D function
        self.line = Line2D(x, y,
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas
        #self.corr_cords = 1

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        #print("Coordinates:", (xyt))
        d = np.hypot(xt - event.x, yt - event.y)
        #print("hypos:",d)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)
        print("Index:", self._ind, "coordinates:", (event.xdata,event.ydata))

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        corr_cord = [event.xdata,event.ydata]
        print("Index:", self._ind, "Corrected coordinates:", corr_cord)
        print("Index from the polylist before correction:",self.poly.xy[self._ind])       ## can update the list over here with the
        corr_cord = [int(i) for i in corr_cord]
        self.poly.xy[self._ind] = corr_cord
        self.corr_cords = self.poly.xy
        print("Index from the polylist after the correction:",self.corr_cords)       ## can update the list over here with the
        self._ind = None
        #return self.corr_cords

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':        # toggle vertex on and off
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':      # delete vertex under the point
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        if self.line.stale:
            self.canvas.draw_idle()

        elif event.key == 'r':
            print("Corrected coordinates are saved in static variable corr_cords:", self.corr_cords)


    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))
        #dict0={}
        #dict0={self._ind:self.poly.xy}
        #print ("Index and the polygon coordinates:",dict0)
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


class Tool(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.image_dir = args.image_dir
        self.data_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)
        self.model.reload(args.reloadmodel, strict=False)

    def get_grid_size(self, run_ggnn=True):
        if self.opts['use_ggnn'] and run_ggnn:
            grid_size = self.model.ggnn.ggnn_grid_size          ## Possibly defines the number of the poly coordinates to be predicted
        else:           ## ggnn_grid_size=112 from the json experiment file
            grid_size = self.model.encoder.feat_size

        return grid_size

    def annotation(self, instance, run_ggnn=False):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)     # an extra dimension is added for the weight matrix
            img = torch.from_numpy(img).to(device)
            # Add batch dimension and make torch Tensor

            output = self.model(
                img,
                poly=None,
                fp_beam_size=5,
                lstm_beam_size=1,
                run_ggnn=run_ggnn
            )
            polys = output['pred_polys'].cpu().numpy()
	    print('polys from the output of the model:',polys)
        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output(polys, instance,
            grid_size)

    def fixing(self, instance, run_ggnn=False):     ## fixing the corrected preds
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            poly = np.expand_dims(instance['corr_cords'], 0)
            img = torch.from_numpy(img).to(device)
            print('Contents inside the fixing method: ', poly)
            poly = torch.from_numpy(poly).to(device)
            # Add batch dimension and make torch Tensor

            output = self.model(
                img,
                poly=poly,
                fp_beam_size=5,
                lstm_beam_size=1,
                run_ggnn=run_ggnn
            )
            polys = output['pred_polys'].cpu().numpy()

        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output(polys, instance,
            grid_size)

    def run_ggnn(self, instance):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            poly = np.expand_dims(instance['fwd_poly'], 0)
            grid_size = self.get_grid_size(run_ggnn=False)
            poly = Utils.xy_to_class(torch.from_numpy(poly), grid_size).numpy()
            img = torch.from_numpy(img).to(device)

            concat_feats, _ = self.model.encoder(img)
            output = self.model.ggnn(
                img,
                poly,
                mode = 'tool',
                resnet_feature = concat_feats
            )
            polys = output['pred_polys'].cpu().numpy()

        del(output)
        grid_size = self.get_grid_size(run_ggnn=True)
        return self.process_output(polys, instance,
            grid_size)

    def process_output(self, polys, instance, grid_size):
        poly = polys[0]
        poly = utils.get_masked_poly(poly, grid_size)
        poly = utils.class_to_xy(poly, grid_size)
        poly = utils.poly0g_to_poly01(poly, grid_size)
        poly = poly * instance['patch_w']
        poly = poly + instance['starting_point']

        torch.cuda.empty_cache()
        return [poly.astype(np.int).tolist()]

class InteractiveGUITest(QMainWindow):
    def __init__(self):
        super(InteractiveGUITest,self).__init__()
        loadUi('Tool/GUIwithPyTorch/InteractiveGUI3.ui',self)
        self.ans = None
        self.image = None
        self.cropped = None
        self.instance = None
        self.pushButton.clicked.connect(self.loadClicked)
        self.ProcessButton.clicked.connect(self.processImageClicked)      ## Loading of the model have to be done here.
        self.PolyButton.clicked.connect(self.getpolyClicked)
    ### Testing the images on the model.

    def InstanceMaker(self, ref_point, fname):    ## Need to define
        instance = {}
        x0, y0, x1, y1 = ref_point[0][0], ref_point[0][1],  ref_point[1][0], ref_point[1][1]
        x1 = x1-x0
        y1 = y1-y0
        bbox = x0,y0,x1,y1
        img_path = fname
        instance = {
        'bbox' : bbox,
        'img_path' : img_path
        }
        return instance


    def loadClicked(self):
            fname,filter = QFileDialog.getOpenFileName(self,'Open File','/home/uib06040/polyrnn',"Image Files (*.png)") ## Image browser
            if fname:
                self.readImage(fname)
            else:
                print('Invalid Image')
            self.TestOnCrop()

    def getpolyClicked(self):
        print('Calculating the corrected coordinates of the polygon now...')


    def processImageClicked(self):     ## Image Cropping:: need to use self.cropped over here.
        #cv2.imwrite("imgs/input.png", self.cropped)
        self.TestOnCrop()

    def save_to_json(crop_name, predictions_dict):
        # To save files to the json annotation about the poly vertices
        output_dict = {'img_source': crop_name, 'polys': predictions_dict['polys'][0].tolist()}
        if 'polys_ggnn' in predictions_dict:
         output_dict['polys_ggnn'] = predictions_dict['polys_ggnn'][0].tolist()
        fname = os.path.basename(crop_name).split('.')[0] + '.json'
        fname = os.path.join(FLAGS.OutputFolder, fname)
        json.dump(output_dict, open(fname, 'w'), indent=4)

    def TestOnCrop(self, run_ggnn=True):
        component = {}
        component['poly'] = np.array([[-1., -1.]])
        img_path = self.instance['img_path']
        self.instance = tool.data_loader.prepare_component(self.instance, component)
        instance = self.instance
        print("Instance after the prepare component::", self.instance)
        pred_annotation = tool.annotation(self.instance, run_ggnn=True)
        stackedpoly = np.column_stack(pred_annotation)
        print("Polygonstack:", stackedpoly)
        poly = Polygon(stackedpoly, animated = True, fill=False) # polygon is drawn here
        #poly.set_facecolor(None)
        fig, ax = plt.subplots()
        img = io.imread(img_path)
        ax.imshow(img, aspect = 'equal', norm = None)
        ax.add_patch(poly)
        p = PolygonInteractor(ax, poly)                          # for the interaction part
        #instance['corr_cords'] = p.cc
        ax.set_title('Click and drag a point to move it')
        #ax.set_xlim((-2, 2))
        #ax.set_ylim((-2, 2))
        print ('Annotations for the instance:', pred_annotation)
        instance['pred_annotations']=pred_annotation
        print('instance:',instance)
        jsons = {
        'img_path' : instance['img_path'],
        'pred_annotations_ggnns' : pred_annotation
        }
        json.dump(jsons, open('annotations.json', 'w'), indent=4)
        plt.show()
        #self.ans = input("Do you want to continue the annotation on the same image:(y/n)")
        #if(self.ans == 'y'):
        #    self.readImage(fname)
        #elif(self.ans == 'n'):
        #    self.loadClicked()
        ##print("Corrected coordinates in the loop:", instance['corr_cords'])

    def readImage(self,fname):  ## Also prepare component here
         self.image = cv2.imread(fname)
         def shape_selection(event, x, y, flags, param):
             # grab references to the global variables
             global ref_point, cropping
             # if the left mouse button was clicked, record the starting
             # (x, y) coordinates and indicate that cropping is being
             # performed
             if event == cv2.EVENT_LBUTTONDOWN:
                 ref_point = [(x, y)]
                 cropping = True

                 # check to see if the left mouse button was released
             elif event == cv2.EVENT_LBUTTONUP:
                 # record the ending (x, y) coordinates and indicate that
                 # the cropping operation is finished
                 ref_point.append((x, y))
                 cropping = False

                 # draw a rectangle around the region of interest
                 cv2.rectangle(images, ref_point[0], ref_point[1], (0, 255, 0), 1) ## Last argument is the line width
                 cv2.imshow("images", images)

         images = self.image # self.image = cv2.imread(fname)
         clone = images.copy()
         cv2.namedWindow("images")
         cv2.setMouseCallback("images", shape_selection)        # shape_selection is looped here
         # keep looping until the 'q' key is pressed
         while True:
           # display the image and wait for a keypress
           cv2.imshow("images", images)
           key = cv2.waitKey(1) & 0xFF

           # if the 'r' key is pressed, reset the cropping region
           if key == ord("r"):
             images = clone.copy()

           # if the 'c' key is pressed, break from the loop
           elif key == ord("c"):
             break

         # if there are two reference points, then crop the region of interest
         # from the image and display it
         if len(ref_point) == 2:
             print("ref_point:", ref_point)

             crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]

             cv2.imshow("crop_img", crop_img)
             cv2.waitKey(0)
             self.cropped = crop_img.copy()
             self.dimcropped = self.cropped.shape
             print("Shape of the cropped Image:",self.dimcropped[0:2])
             dim = (224,224)
             resized = cv2.resize(self.cropped, dim, interpolation = cv2.INTER_AREA)
             self.cropped = resized.copy() ## here contains the cropped image in 224*224
             print("Shape of the resized Image:",self.cropped.shape)

         instance = self.InstanceMaker(ref_point,fname)
         self.instance = instance
         print('Instance dictionary:', self.instance)
         # close all open windows
         cv2.destroyAllWindows()
         self.displayCroppedImage()

    def displayCroppedImage(self, window =1):
            qformat = QImage.Format_Indexed8
            if len(self.cropped.shape) == 3: # rows[0],cols[1],channels[2]
                if(self.cropped.shape[2]) == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.cropped, self.cropped.shape[1],self.cropped.shape[0],self.cropped.strides[0],qformat)

            #BGR > RGB
            img = img.rgbSwapped()
            if window == 1:
                self.imglabel.setPixmap(QPixmap.fromImage(img))
                self.imglabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            if window == 2:
                qformat2 = QImage.Format_Indexed8
                if len(self.croppedGGNN.shape) == 3: # rows[0],cols[1],channels[2]
                    if(self.croppedGGNN.shape[2]) == 4:
                        qformat2 = QImage.Format_RGBA8888
                    else:
                        qformat2 = QImage.Format_RGB888

                self.imglabel_2.setPixmap(QPixmap.fromImage(img))
                self.imglabel_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
                img2 = QImage(self.croppedGGNN, self.croppedGGNN.shape[1],self.croppedGGNN.shape[0],self.croppedGGNN.strides[0],qformat2)
                img2 = img2.rgbSwapped()
                x,y = self.croppedGGNN.shape[0], self.croppedGGNN.shape[1]
                print("X and Y :",self.croppedGGNN.shape)
                imcrop = self.croppedGGNN[45:329,48+2:332]      ## Complicated hardcoded numbers to overlay the image back to the orginal one.
                resized = cv2.resize(imcrop, (self.dimcropped[1]-2,self.dimcropped[0]-2), interpolation = cv2.INTER_AREA)
                y_offset, x_offset, y_offset1,  x_offset1 = ref_point[0][0]+2,ref_point[0][1]+2, ref_point[1][0], ref_point[1][1]
                print("X and Y offset:",x_offset, y_offset, x_offset1, y_offset1 )
                self.image[x_offset:x_offset1, y_offset:y_offset1 ] = resized
                cv2.imwrite("overlay/input.png", self.image)
                self.imglabel_3.setPixmap(QPixmap.fromImage(img2))
                self.imglabel_3.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    args = get_args()
    global tool
    tool = Tool(args)
    window=InteractiveGUITest()
    window.setWindowTitle('GUI for polyrnn++')
    window.show()
    sys.exit(app.exec_())
