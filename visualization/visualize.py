#-*- coding: utf-8 -*-
# visualization code for sgcls task
# in KERN root dir, run python visualization/visualize_sgcls.py -cache_dir caches/kern_sgcls.pkl -save_dir visualization/saves
from dataloaders.visual_genome import VCRDataset,VGDataLoader, VG, vg_collate
from graphviz import Digraph
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
os.makedirs(args.save_dir, exist_ok=True)
image_dir = os.path.join(args.save_dir, 'images')
graph_dir = os.path.join(args.save_dir, 'graphs')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)
mode = 'sgdet' # this code is only for sgcls task

train, _, _ = VG.splits(num_val_im=5000, filter_duplicate_rels=True,
                        use_proposals=False,
                        filter_non_overlap=False)
'''

train,_, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                         use_proposals=conf.use_proposals,
                         filter_non_overlap=conf.mode == 'sgdet')
'''
vcrdata = VCRDataset()
vcrdataloader = DataLoader(vcrdata, batch_size=1, shuffle=True,
           batch_sampler=None, num_workers=1, collate_fn=lambda x: vg_collate(x, mode='rel', num_gpus=1, is_train=False), drop_last=True)


ind_to_predicates = train.ind_to_predicates
ind_to_classes = train.ind_to_classes


def visualize_pred(fn, pred_entry, ind_to_classes, ind_to_predicates, image_dir, graph_dir, save_format='jpg'): #save_format='pdf'):
    #im = mpimg.imread(os.path.join('/home/suji/spring20/vilbert_beta/data/VCR/vcr1images/lsmdc_1049_Harry_Potter_and_the_chamber_of_secrets', fn))
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
    '''
    print('len roi:',len(rois))
    print('len pred_classes_str:', len(pred_classes))
    print("len(pred_rel_inds):", len(pred_rel_inds))
    print("(pred_rel_inds):", (pred_rel_inds))
    '''
    sg_save_fn = os.path.join(graph_dir, "_".join([fn.split('/')[-2], os.path.basename(fn)[:-4]+'.'+save_format]))
    u = Digraph('sg', filename=sg_save_fn, format=save_format)
    u.attr('node', shape='box')
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')
    name_list_pred = []
    for i,l in enumerate(pred_classes):
        #print(i)
        name_pred = ind_to_classes[pred_classes[i]]
        name_suffix_pred = 1
        obj_name_pred = name_pred

        while obj_name_pred in name_list_pred:
            obj_name_pred = name_pred + '_' + str(name_suffix_pred)
            name_suffix_pred += 1
        name_list_pred.append(obj_name_pred)
        #print('obj_name_pred:', obj_name_pred)
        u.node(str(i), label=obj_name_pred, color='forestgreen')
    pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
    keep_r = set()
    for row_no,( ind, score_list) in enumerate(zip(rel_scores[:,1:].argmax(1), rel_scores)):
        if (score_list[ind]> .01):
            keep_r.add(row_no)
            print('rel_score:', score_list[ind])
    '''
    print('pred_rel:', len(pred_rels))
    print('keep_r len:', len(keep_r))
    '''
    pred_rels = np.take(pred_rels, list(keep_r), axis = 0)
    #print('pred_rel:', len(pred_rels))
    #pred_rels = pred_rels[:top_k]
    obj_inds = set()
    for rel in pred_rels:
        obj_inds.add(rel[0])
        obj_inds.add(rel[1])
        u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[rel[2]], color='forestgreen')
    u.render(view=False, cleanup=True)
    for i, bbox in enumerate(rois):
        #if i not in obj_inds:
        #    continue
        pred_classes_str = ind_to_classes[int(pred_classes[i])]

        while pred_classes_str in object_name_list_pred:
            obj_count_pred[int(pred_classes[i])] += 1
            pred_classes_str = pred_classes_str + '_' + str(obj_count_pred[int(pred_classes[i])])
        object_name_list_pred.append(pred_classes_str)
        # if labels[i] == pred_classes[i]:
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
    #image_save_fn = os.path.join(image_dir, fn.split('/')[-1].split('.')[-2]+'.'+save_format)
    image_save_fn = os.path.join(image_dir,"_".join([fn.split('/')[-2], os.path.basename(fn)[:-4]+'.'+save_format]))
    plt.savefig(image_save_fn)
    plt.close()


with open(args.cache_dir, 'rb') as f:
    all_pred_entries = pkl.load(f)
print("pred entries loaded")
for i, pred_entry in enumerate(tqdm(all_pred_entries)):
    print("my fn", vcrdata.filenames[i])
    fn = pred_entry['fn']#vcrdata.filenames[i]
    print("other fn", fn)
    visualize_pred(fn, pred_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir)

