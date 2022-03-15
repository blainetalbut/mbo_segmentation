import image_graph
import mbo
import PIL, PIL.Image
import numpy as np

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

    mask.show()
    return mask
    # print(f'mask = {mask}')
    # masked_img = PIL.Image.composite(PIL.Image.new('L', img.size), img, mask)
    # masked_img.show()

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

test_image_f = "twocows.bmp"
test_image_gt_f = "twocows_GT.bmp"

test_image = PIL.Image.open(test_image_f).convert("L")
test_image_gt = PIL.Image.open(test_image_gt_f)
gt = np.apply_along_axis(gtmap, axis=2, arr=np.array(test_image_gt)).flatten()

test_img_g = image_graph.Img_Graph([test_image])

def make_graph():
    # print("About to generate the graph.")
    g = test_img_g.neighborhood_feature_graph(patch_size=7, num_nbr=100, sigma=100)
    return g

def make_seggr(g=None):
    if g==None:
        g = make_graph()
    seg_args = {
            'g':g,
            'parr0' : gt,
            'num_eigenvecs' : 200,
            'C' : 30,
            'dt' : 0.003,
            'stopcond' : 1e-7
            }

    seggr = mbo.MBO_segmenter( **seg_args )
    return seggr

def show_eigenvecs(seggr=None):
    if not seggr:
        seggr = make_seggr()
    vs = seggr.laplacian_spectrum()[0]
    for k in range(vs.shape[1]):
        if k%20==0:
            show_mask(test_image, vs[:,k])

# u = seggr.run()
# print('u={}'.format(u))
# print(f'error = {sum(u==gt)/len(gt)}')
                
# test_image_gt.show()
# mask = show_mask(test_image, u)


