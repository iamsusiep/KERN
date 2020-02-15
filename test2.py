from PyPDF2 import PdfFileMerger, PdfFileReader

from PIL import Image

import os
from glob import glob 



def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


