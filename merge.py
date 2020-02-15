import os
from glob import glob
from PyPDF2 import PdfFileWriter, PdfFileReader
def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()
            
    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
        
    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)
        
        
if __name__ == '__main__':
    #paths = glob.glob('w9_*.pdf')
    #paths.sort()
    for fn in glob('visualization/vcr_saves/images/*'):
        fn2 = 'visualization/vcr_saves/graphs/{}.pdf'.format(os.path.basename(fn))
        paths = [fn, fn2]
        out_fn = 'concat/{}'.format(os.path.basename(fn))
        print(out_fn)
        merger(out_fn, paths)
     
