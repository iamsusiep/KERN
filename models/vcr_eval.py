from dataloaders.visual_genome import VGDataLoader, VG, VCRDataset, vg_collate
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
from lib.kern_model import KERN
from torch.utils.data import DataLoader 

conf = ModelConfig()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
vcrdata = VCRDataset()

vcrdataloader = DataLoader(vcrdata, batch_size=conf.batch_size * conf.num_gpus, shuffle=True,
           batch_sampler=None, num_workers=conf.num_workers, collate_fn=lambda x: vg_collate(x, mode='rel', num_gpus=conf.num_gpus, is_train=False), drop_last=True)

detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
                ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
                use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
                use_ggnn_rel=conf.use_ggnn_rel, ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
                ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim, ggnn_rel_output_dim=conf.ggnn_rel_output_dim,
                use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge)

detector.cuda()
ckpt = torch.load(conf.ckpt)
all_pred_entries= []
optimistic_restore(detector, ckpt['state_dict'])
def val_batch(batch_num, b, thrs=(20, 50, 100)):
    print('val_batch, type(b)',type(b))
    print('b.get_fns()', b.get_fns())
    assert len(b.get_fns()) == 1
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]
    fn = b.get_fns()[0]
    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
        #print("entry", entry[0])
        #print("type entry", type(entry))
        print('type fn',type(fn))
        print(fn)
        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
            'fn': fn
        }
        all_pred_entries.append(pred_entry)

detector.eval()
for val_b, batch in enumerate(tqdm(vcrdataloader)):
    val_batch(conf.num_gpus*val_b, batch)

if conf.cache is not None:
    with open(conf.cache,'wb') as f:
        pkl.dump(all_pred_entries, f)
