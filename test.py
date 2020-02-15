import time
import os
from glob import glob 
from pdf2jpg import pdf2jpg
start_time = time.time()
#for input_path in glob("concat_0.001/*"):
for input_path in glob("concat/*"):
    output_path = "test/"
    result = pdf2jpg.convert_pdf2jpg(input_path, output_path, dpi=50, pages="ALL")
print("Completed: ", time.time() - start_time)
'''
for fn in glob('visualization/vcr_saves/images/*'):
    fn2 = 'visualization/vcr_saves/graphs/{}'.format(os.path.basename(fn))
    print('f', fn)
    print('f2', fn2)
    out_fn = 'test/{}'.format(os.path.basename(fn))
    out_fn2 = 'test/2{}'.format(os.path.basename(fn))
    print('o',out_fn)
    if not os.path.exists(os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn), exist_ok = True)
    result = pdf2jpg.convert_pdf2jpg(fn, out_fn, pages="1")
    result2 = pdf2jpg.convert_pdf2jpg(fn2, out_fn2, pages="1")
    break;
'''
