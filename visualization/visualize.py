#-*- coding: utf-8 -*-
# in KERN root dir, run python visualization/visualize_sgcls.py -cache_dir caches/kern_sgcls.pkl -save_dir visualization/saves
from dataloaders.visual_genome import VCRDataset,VGDataLoader, VG, vg_collate
from graphviz import Digraph
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from config import ModelConfig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dill as pkl
from collections import defaultdict
import gc 
import os
# conf = ModelConfig()
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
import argparse
from torch.utils.data import DataLoader 
import os
from glob import glob
from PyPDF2 import PdfFileWriter, PdfFileReader

# relation threshold
rel_thres = 0.005

def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()
            
    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
        
    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)
        
parser = argparse.ArgumentParser(description='visualization for sgcls task')
parser.add_argument(
    '-save_dir',
    dest='save_dir',
    help='dir to save visualization files',
    type=str,
    default='visualization/saves'
)

parser.add_argument(
    '-cache_dir',
    dest='cache_dir',
    help='dir to load cache predicted results',
    type=str,
    default='caches/kern_sgcls.pkl'
)


args = parser.parse_args()
mode = 'sgdet' # this code is only for sgcls task

train, _, _ = VG.splits(num_val_im=5000, filter_duplicate_rels=True,
                        use_proposals=False,
                        filter_non_overlap=False)
vcrdata = VCRDataset()
vcrdataloader = DataLoader(vcrdata, batch_size=1, shuffle=False,
           batch_sampler=None, num_workers=1, collate_fn=lambda x: vg_collate(x, mode='rel', num_gpus=1, is_train=False), drop_last=True)


ind_to_predicates = train.ind_to_predicates
ind_to_classes = train.ind_to_classes

# Directory strucutre: vis_{threshold}/moviename/imagename without file extension/graph.png or /image.png
def visualize_pred(fn, pred_entry, ind_to_classes, ind_to_predicates, image_dir= None, graph_dir=None, save_format='png'):
    im = mpimg.imread(fn)
    max_len = max(im.shape)
    scale = BOX_SCALE / max_len
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(im, aspect='equal')
    rois = pred_entry['pred_boxes']
    rois = rois / scale

    pred_classes = pred_entry['pred_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
    object_name_list = []
    obj_count = np.zeros(151, dtype=np.int32)
    object_name_list_pred = []
    obj_count_pred = np.zeros(151, dtype=np.int32)

    sg_save_fn = "vis_{}/".format(rel_thres) + "/".join((fn.replace('.jpg', '')).split("/")[-2:] + ["graph"])
    dir_name = os.path.dirname(sg_save_fn)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    #sg_save_fn = os.path.join(graph_dir, "_".join([fn.split('/')[-2], os.path.basename(fn).replace('jpg', save_format)]))
    u = Digraph('sg', filename=sg_save_fn, format=save_format)
    u.attr('node', shape='box')
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')
    name_list_pred = []
    for i,l in enumerate(pred_classes):
        name_pred = ind_to_classes[pred_classes[i]]
        name_suffix_pred = 1
        obj_name_pred = name_pred

        while obj_name_pred in name_list_pred:
            obj_name_pred = name_pred + '_' + str(name_suffix_pred)
            name_suffix_pred += 1
        name_list_pred.append(obj_name_pred)
        u.node(str(i), label=obj_name_pred, color='forestgreen')
    pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
    keep_r = set()
    for row_no,( ind, score_list) in enumerate(zip(rel_scores[:,1:].argmax(1), rel_scores)):
        if (score_list[ind]> rel_thres):
            keep_r.add(row_no)
            print('rel_score:', score_list[ind])
    pred_rels = np.take(pred_rels, list(keep_r), axis = 0)
    for rel in pred_rels:
        u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[rel[2]], color='forestgreen')
    u.render(view=False, cleanup=True)
    for i, bbox in enumerate(rois):
        pred_classes_str = ind_to_classes[int(pred_classes[i])]

        while pred_classes_str in object_name_list_pred:
            obj_count_pred[int(pred_classes[i])] += 1
            pred_classes_str = pred_classes_str + '_' + str(obj_count_pred[int(pred_classes[i])])
        object_name_list_pred.append(pred_classes_str)
        ax.text(bbox[0], bbox[1] - 2,
                pred_classes_str,
                bbox=dict(facecolor='green', alpha=0.5),
                fontsize=32, color='white')
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='green', linewidth=3.5)
        )

    ax.axis('off')
    fig.tight_layout()
    #image_save_fn = os.path.join(image_dir,"_".join([fn.split('/')[-2], os.path.basename(fn)]))
    image_save_fn = sg_save_fn.replace("graph", "image") + ".{}".format(save_format) 
    plt.savefig(image_save_fn)
    plt.close()


with open(args.cache_dir, 'rb') as f:
    all_pred_entries = pkl.load(f)

for i, pred_entry in enumerate(tqdm(all_pred_entries)):
    fn = pred_entry['fn']#vcrdata.filenames[i]
    visualize_pred(fn, pred_entry, ind_to_classes, ind_to_predicates)
for dir_name in glob('vis_{}/*/*/'.format(str(rel_thres))):
    print('read from',dir_name)
    img_fn = dir_name + "image.png"
    graph_fn = dir_name + "graph.png"
    img_image, graph_image = [Image.open(x) for x in [img_fn, graph_fn]]
    if img_image.size[1] > graph_image.size[1]:
        ratio = graph_image.size[1]/img_image.size[1]
        if ratio >0.5:
            new_width, new_height = ratio * img_image.size[0], ratio * img_image.size[1] #resize image
            img_image.thumbnail((new_width, new_height))
    images = [img_image, graph_image]#[Image.open(x) for x in [img_fn, graph_fn]]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    out_fn = (dir_name.replace("vis", "concat_vis")[:-1] + ".jpg")
    print('out:',out_fn)
    new_dirname = os.path.dirname(out_fn)
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname, exist_ok=True)
    new_im.save(out_fn)
'''
for fn in glob('visualization/vcr_saves_{}/images/*'.format(str(rel_thres))):
    fn2 = 'visualization/vcr_saves_{}/graphs/{}.pdf'.format(str(rel_thres), os.path.basename(fn))
    paths = [fn, fn2]
    out_fn = 'concat_{}/{}'.format(str(rel_thres), os.path.basename(fn))
    print(out_fn)
    if (os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn), exist_ok = True)
    merger(out_fn, paths)
'''
