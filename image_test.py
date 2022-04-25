import image_graph
import mbo
import PIL, PIL.Image
import numpy as np
import random
import math

def renormalize(u):
    m = abs(min(u))
    M = abs(max(u))
    return u / max(m,M)


def show_mask(img, u):
    # potential returned by mbo segmenter is not in uint [0,255] format
    u = renormalize(u) 
    shape = img.size[1], img.size[0] # the shape of the array is the reverse of the shape of the image; opposite conventions
    mask_ar = ( u * (255/2) + (255/2) ).astype(np.uint8)
    mask = PIL.Image.fromarray(np.reshape(mask_ar, shape))
    canvas = PIL.Image.new(img.mode, img.size)

    masked_img = PIL.Image.composite(img, canvas, mask)
    masked_img.show()

    return masked_img

def make_potential(g, gt = None):
    return mbo.MBO_segmenter(
            g = g,
            parr0 = gt,
            num_eigenvecs = 30,
            dt = .01,
            C = 1,
            stopcond = 1e-7 ).run()

def gtmap(a):
    if a[2] == 128:
        return 1
    else:
        return -1

def save_gif(ims, filename, duration=500):
    ims[0].save('gifs/' + filename, save_all=True, append_images=ims[1:], duration=duration)

def pseudo_threshold(u, percentile=75):
    k = np.percentile(u**2, percentile)
    v = u/ math.sqrt(k)
    v = np.minimum(v, [1]*len(v))
    v = np.maximum(v, [-1]*len(v))
    return v

def train_filter(gt, sample_rate=0.01):
    l = len(gt)
    k = int(l*sample_rate)
    sample_filter = [0] * l
    for i in random.choices(range(l), k=k):
        sample_filter[i] = 1
    return gt * sample_filter

train_image_f = "twocows.bmp"
train_image_gt_f = "twocows_GT.bmp"

test_image_f = "./twocows_1_21.bmp"
test_image_gt_f = "./twocows_1_21_GT.bmp"

train_image = PIL.Image.open(train_image_f)
test_image = PIL.Image.open(test_image_f)

train_image_gt_img = PIL.Image.open(train_image_gt_f)
train_image_gt = np.apply_along_axis(gtmap, axis=2, arr=np.array(train_image_gt_img)).flatten()
test_image_gt = np.apply_along_axis(gtmap, axis=2, arr=np.array(PIL.Image.open(test_image_gt_f))).flatten()

# smol images for rapid testing
smol_size = (32,21)
smol_train_image = train_image.resize(smol_size)
smol_test_image = test_image.resize(smol_size)
smol_train_image_gt_img = train_image_gt_img.resize(smol_size)
smol_train_image_gt = np.apply_along_axis(gtmap, axis=2, arr=np.array(smol_train_image_gt_img)).flatten()
smol_imgs_g = image_graph.Img_Graph([smol_train_image, smol_test_image])
smol_supervision = np.append(smol_train_image_gt, np.array([0] * smol_train_image_gt.size))

l = smol_train_image_gt.size
sample_filter = [0] * l
sample_rate = 0.01
for i in random.choices(range(l), k=int(l*sample_rate)):
    sample_filter[i] = 1
smol_sampled_supervision = smol_train_image_gt * sample_filter

ig = image_graph.Img_Graph([train_image])
one_image_supervision = train_filter(train_image_gt)

# transferring a label from one image to another doesn't work well when the labeling is BINARY
# TO TRY: multivalued labels
imgs_g = image_graph.Img_Graph([train_image, test_image])
supervision = np.append(train_image_gt, np.array([0.5] * test_image_gt.size))

def make_graph(ig = imgs_g, patch_size=7, num_nbr=100, sigma=100):
    return ig.neighborhood_feature_graph(patch_size = patch_size, num_nbr = num_nbr, sigma=sigma)

def make_seggr(g=None, supervision=supervision):
    if g==None:
        g = make_graph()
    seg_args = {
            'g':g,
            'parr0' : supervision,
            'num_eigenvecs' : 200,
            'C' : 30,
            'dt' : 0.003,
            'stopcond' : 1e-7
            }

    seggr = mbo.MBO_segmenter(**seg_args)
    return seggr

def show_eigenvecs(seggr=None, ig = imgs_g):
    if not seggr:
        seggr = make_seggr()
    vs = seggr.laplacian_spectrum()[0]
    for k in range(vs.shape[1]):
        if k%20==0:
            ig.show_mask(vs[:,k])

def smol_test():
    g = make_graph(ig = smol_imgs_g, patch_size=3, num_nbr=10, sigma=10)
    seggr = mbo.MBO_segmenter(g=g, parr0=smol_supervision, num_eigenvecs=200, C=30, dt=0.003, stopcond=1e-7)
    u = seggr.run()
    smol_imgs_g.show_mask(u)
    return seggr,u

def single_smol_test():
    ig = image_graph.Img_Graph([smol_train_image])
    g = make_graph(ig=ig, patch_size=3, num_nbr=100, sigma=10)
    seggr = mbo.MBO_segmenter(g=g, parr0=train_filter(smol_train_image_gt, sample_rate=0.4), num_eigenvecs=200, C=30, dt=0.003, stopcond=1e-7)
    u = seggr.run()
    ig.show_mask(u)
    return ig, seggr, u

# u = seggr.run()
# print('u={}'.format(u))
# print(f'error = {sum(u==gt)/len(gt)}')
                
# test_image_gt.show()
# mask = show_mask(test_image, u)


