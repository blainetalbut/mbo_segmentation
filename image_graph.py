# import an image and convert it to a graph representation (similarity matrix of pixels)

from PIL import Image
import numpy as np
import math
import sklearn
import sklearn.neighbors
import scipy

class Img_Graph:
    def __init__(self, imgs):
        self._imgars = []
        for img in imgs:
            try: assert(img.mode == 'L')
            except(AssertionError): raise ValueError("images must be greyscale")
            self._imgars.append(np.asarray(img)) # store the images as 2d arrays
        self.size = sum([ar.size for ar in self._imgars])
        self.g = None
        self.features = None
        self.dist_g = None

    def neighborhood_feature_graph(self, patch_size=7, num_nbr=100, sigma=10):
        assert(patch_size%2==1)
        size = sum([ar.size for ar in self._imgars])
        features = [None]*size # array to store feature vectors (patches)

        running_tail = 0
        for indx, ar in enumerate(self._imgars):
            height, width = ar.shape
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

    # u: an assignment of pixels to communities
    # #    an array of size (total size of all imagrs)
    def masked_imgs(self, u):
        imgs = []
        running_tail = 0
        for imgar in self._imgars:
            img = Image.fromarray(imgar)
            mask_ar = u[running_tail:imgar.size]
            mask_ar = np.reshape(mask_ar, imgar.shape)
            mask = Image.fromarray(mask_ar)

            imgs.append( Image.composite(img, Image.new(mode='L', size=imgar.shape), mask) )
            running_tail += img.size

        return imgs


# test_image_file = "./GraphAlgorithms/data/Images/twocows.png"
# test_img = Image.open(test_image_file).convert("L")
# test_img_g = Img_Graph([test_img])
# fg = test_img_g.neighborhood_feature_graph()
