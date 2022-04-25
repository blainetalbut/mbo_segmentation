# import an image and convert it to a graph representation (similarity matrix of pixels)

from PIL import Image
import numpy as np
import math
import sklearn
import sklearn.neighbors
import scipy

def renormalize(u):
    m = abs(min(u))
    M = abs(max(u))
    return u / max(m,M)


def mask(img, u):
    # potential returned by mbo segmenter is not in uint [0,255] format
    u = renormalize(u) 
    shape = img.size[1], img.size[0] # the shape of the array is the reverse of the shape of the image; opposite conventions
    mask_ar = ( u * (255/2) + (255/2) ).astype(np.uint8)
    mask = Image.fromarray(np.reshape(mask_ar, shape))
    canvas = Image.new(img.mode, img.size)

    masked_img = Image.composite(img, canvas, mask)

    return masked_img

class Img_Graph:
    def __init__(self, imgs):
        self._imgars = []
        self.mode = imgs[0].mode
        for img in imgs:
            assert(img.mode==self.mode)
            self._imgars.append(np.array(img)) # store the images as 2d arrays
        self.size = sum([ar.size for ar in self._imgars])
        self.g = None
        self.features = None
        self.dist_g = None

        
    def _neighborhood_feature_graph_rgb(self, patch_size=7, num_nbr=100, sigma=10):
        assert(patch_size%2==1)
        size = sum([ar.shape[0]*ar.shape[1] for ar in self._imgars])
        features = [None]*size # array to store feature vectors (patches): length is total number of pixels across all images

        running_tail = 0
        for indx, ar in enumerate(self._imgars):
            height, width, _ = ar.shape
            padding = int((patch_size-1)/2)
            pad_widths = ((padding,),(padding,),(0,))
            local_ar = np.pad(ar,pad_widths) # padding preserves shape after we take patches

            for i in range(height):
                for j in range(width): # collect patches
                    v = local_ar[i:i+patch_size, j:j+patch_size, :].flatten() 
                    index = running_tail + i*width + j
                    try: 
                        features[index] = v
                    except: 
                        raise IndexError(f"Feature assignment index {index} out of range. runnning_tail = {running_tail}. i={i}. j={j}.")

            running_tail += height*width
        features = np.array(features)
        # by default, PIL.Image arrays are stored as 8-bit uints, causing frequent overflow errors when we compute Assignmentidistances
        features = features.astype(float)         
        print(features)
        print(features.shape)

        nbr_graph = sklearn.neighbors.kneighbors_graph(features, num_nbr, mode='distance', include_self=True)
        print("kgraph made")
        print(f'graph shape = {nbr_graph.shape}')
        # weights = np.sum(features, axis=1)
        # nbr_graph = nbr_graph / (weights @ weights.T)

        self.dist_g = nbr_graph.copy()

        nbr_graph.data = np.exp(-0.5 * nbr_graph.data ** 2 / sigma ** 2)

        self.g = nbr_graph
        self.features = features
        return self.g

    def _neighborhood_feature_graph_bw(self, patch_size=7, num_nbr=100, sigma=10):
        assert(patch_size%2==1)
        size = sum([ar.size for ar in self._imgars])
        features = [None]*size # array to store feature vectors (patches)

        running_tail = 0
        for indx, ar in enumerate(self._imgars):
            height, width  = ar.shape
            padding = int((patch_size-1)/2)
            local_ar = np.pad(ar,padding) # padding preserves shape after we take patches

            for i in range(height):
                for j in range(width): # collect patches
                    v = local_ar[i:i+patch_size, j:j+patch_size].flatten() 
                    features[running_tail + i*width + j] = v

            running_tail += ar.size 
        features = np.array(features)
        # by default, PIL.Image arrays are stored as 8-bit uints, causing frequent overflow errors when we computedistances
        features = features.astype(float)         
        print(features)
        print(features.shape)

        nbr_graph = sklearn.neighbors.kneighbors_graph(features, num_nbr, mode='distance', include_self=True)
        print("kgraph made")
        print(f'graph shape = {nbr_graph.shape}')
        # weights = np.sum(features, axis=1)
        # nbr_graph = nbr_graph / (weights @ weights.T)

        self.dist_g = nbr_graph.copy()

        nbr_graph.data = np.exp(-0.5 * nbr_graph.data ** 2 / sigma ** 2)

        self.g = nbr_graph
        self.features = features
        return self.g

    def neighborhood_feature_graph(self, **kwargs):
        if self.mode == 'L':
            return self._neighborhood_feature_graph_bw(**kwargs)
        if self.mode == 'RGB':
            return self._neighborhood_feature_graph_rgb(**kwargs)
            
    # u: an assignment of pixels to communities
    # #    an array of size (total size of all imagrs)
    def masked_imgs(self, u):
        masked_imgs = []
        running_tail = 0
        for imgar in self._imgars:
            img = Image.fromarray(imgar)
            size = imgar.shape[0]*imgar.shape[1]
            mask_ar = u[running_tail:running_tail+size]

            masked_img = mask(img, mask_ar)
            masked_imgs.append(mask(img,mask_ar))

            running_tail += size

        return masked_imgs

    def show_mask(self,u):
        masked_imgs = self.masked_imgs(u)
        for img in masked_imgs:
            img.show()
