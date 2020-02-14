from PIL import Image

import os
from glob import glob 

#im1 = Image.open('data/src/lena.jpg')
#im2 = Image.open('data/src/rocket.jpg')
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

for fn in glob('visualization/vcr_saves/images'):
    fn2 = 'visualization/vcr_saves/graphs/{}'.format(os.path.basename(fn))
    out_fn = 'concat/'.format(os.path.basename(fn))
    get_concat_h(fn, fn2).save(out_fn)
    #for j in glob('visualization/vcr_saves/graphs')

#get_concat_h(im1, im1).save('data/dst/pillow_concat_h.jpg')
#get_concat_v(im1, im1).save('data/dst/pillow_concat_v.jpg')
