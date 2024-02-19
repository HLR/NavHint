import json
from re import L, sub
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
import spacy
nlp = spacy.load("en_core_web_sm")

import numpy as np
from pytorch_transformers import (BertConfig, BertTokenizer)

tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

#/localscratch/zhan1624/VLN-interactive/data/aug_sub.json
#/localscratch/zhan1624/CLIP_prefix_caption/data_generation/sub_instr/sub_train.json
data_path = "/localscratch/zhan1624/CLIP_prefix_caption/data_generation/sub_instr/sub_train.json"
candi_path  = "/egr/research-hlr/joslin/candidate.npy"
object_path = "/localscratch/zhan1624/CLIP_prefix_caption/data_generation/all_output.json"
reason_path = "/localscratch/zhan1624/CLIP_prefix_caption/data_generation/reason_text_v3/"

candi_info = np.load(candi_path, allow_pickle=True).item()

with open(data_path) as f_in1, open(object_path) as f_in2:
    data = json.load(f_in1)
    objects = json.load(f_in2)


def check_direction(next_heading, heading):
    direct_key = ""
    if next_heading - heading > 3 and next_heading - heading < 11:
        direct_key = "right"
    elif  next_heading - heading < -3 and next_heading - heading > -11:
        direct_key = "left"
    elif next_heading - heading < -11:
        direct_key = "down"
    elif next_heading - heading > 11:
        direct_key = "up"
    else:
        direct_key = "front"
    return direct_key

def check_landmark(object_land, sub_instr):
    new_object_land = []
    original_land = []
    res = []
    for o_l in object_land:
        if "," in o_l:
            o_l = o_l.split(',')
            new_object_land += o_l
            original_land += o_l
        else:
            new_object_land.append(nlp(o_l)[-1].lemma_)
            original_land.append(o_l)
    sub_instr = nlp(sub_instr)
    sub_tok = []
    for tok in sub_instr:
        sub_tok.append(tok.lemma_)

    for ob_id, n_o_l in enumerate(new_object_land):
        if n_o_l in sub_tok:
            res.append(original_land[ob_id])
    return res


def start_reason(scan, start_view, target_view, sub_instr, exact_same=False):
    
    candidates = candi_info[scan][start_view]
    repeted = []
    sub_instr = sub_instr.strip()

    ### obtain target viewpoint
    for each_c in candidates:
        if each_c["viewpointId"] == start_view:
            continue
        pointId = each_c["pointId"]
        object_land = objects[scan+"_"+start_view+"_"+str(pointId)]
        if each_c["viewpointId"] == target_view:
            target_obj = object_land
            break
            
    for each_c in candidates:
        if each_c["viewpointId"] == start_view:
            continue
        pointId = each_c["pointId"]
        object_land = objects[scan+"_"+start_view+"_"+str(pointId)]
        if each_c["viewpointId"] == target_view:
            # check direction
            #direct_key = check_direction(pointId, each_c['pointId'])
            # check landmark overlap
            check_land_list = check_landmark(object_land, sub_instr)
            
        else: 
            if object_land == target_obj:
                exact_same = True
            else:
                repeted += object_land
    non_distinct = target_obj.copy()
    overlap_land = list(set(check_land_list).intersection(set(repeted))) # overlapping between landmark and all_candidates
    overlap_can = list(set(target_obj).intersection(set(repeted)))
    for each_ol in overlap_land:
        target_obj.remove(each_ol)
        check_land_list.remove(each_ol)
        overlap_can.remove(each_ol)
    for o_c in overlap_can:
        target_obj.remove(o_c)
 
    return overlap_land, target_obj, check_land_list, non_distinct, exact_same

            

def reason_text_gen(sub_instr, overlap, left_land, overlap_res, non_distinct, exact_same=None):
    """
    overlap: overlapped landmark between objects and landmarks
    left_land: except overlap, resting objects can be seen.
    overlap_res: except overlap, resting objects in sub_instruction
    """
    sub_text = "The instruction \"%s\" need to be executed." % sub_instr
    #dir_text = "The target viewpoint is on the %s." % dir
    target_text1 = ""
    target_text2 = ""
    if overlap:
        # non-distinctive
        target_text1 = "The landmarks \"%s\" in the instruction can be observed in multiple viewpoint." % ",".join(overlap)
        if len(left_land) > 0:
            if overlap_res:
                target_text2 = "But, the landmarks \"%s\" in the instruction are distinctive objects." % ",".join(overlap_res)
            else:
                target_text2 = "However, the distinctive objects \"%s\" maybe helpful." % ",".join(left_land)
        else:
            if exact_same:
                target_text2 += "There are exactly same viewpoints"
            else:
                target_text2 += "There are no other distinguished objects."
    else:
        if overlap_res:
            target_text2 = "The landmarks \"%s\" in the instruction are distinctive objects." % ",".join(overlap_res)
        else:
            if left_land:
                target_text2 = "All landmarks in the instruction are not observed, but the distinctive objects \"%s\" maybe helpful." % ",".join(left_land)
                if exact_same:
                    target_text2 += "There are exactly same viewpoints."
            else:
                if non_distinct:
                    target_text2 += "There are no distinctive objects, but the non-distinctive objects \"%s\" maybe helpful." % ",".join(non_distinct)
                else:
                    target_text2 += "There are no distinctive objects"
    all_text = sub_text + " " + " "+ target_text1 + " "+ target_text2
    return all_text


if 'aug' not in data_path:
    for item in tqdm(data[:20]):
        path = item['path']
        heading = item['heading']
        scan = item['scan']
        reason_text = []
        for j, instr in enumerate(item['instructions']):
            if len(item['instructions']) != len(item['chunk_view']):
                continue
            tmp_reason = {}
            for view_id, each_view in enumerate(item['chunk_view'][j]):
                start  = each_view[0]-1
                end = each_view[1] - 1
                sub_instr = item['sub_instr'][j][view_id]
                views = [_ for _ in range(start, end+1)]
                while end > start:
                    start_view = path[start]
                    target_view = path[start+1]
                    overlap_land, left_land, check_land, non_distinct, exact_same = start_reason(scan, start_view, target_view, sub_instr)
                    text = reason_text_gen(sub_instr, overlap_land, left_land, check_land,  non_distinct, exact_same)
                    tmp_reason[start_view+"_"+target_view] = text
                    start += 1
            reason_text.append(tmp_reason)
        item['reason_text'] = reason_text

else:
    for item in tqdm(data):
        path = item['path']
        heading = item['heading']
        scan = item['scan']
        tmp_reason = {}
        for view_id, (key, value) in enumerate(item['split_target'].items()):
            if view_id == len(path)-1:
                continue
            start_index, end_index = value[0]+1, value[1]+1
            sub_instr = tokenizer.decode(item['instr_enc'][start_index: end_index])
            start_view = key
            target_view = path[view_id+1]
            overlap_land, left_land, check_land, non_distinct, exact_same = start_reason(scan, start_view, target_view, sub_instr)
            text = reason_text_gen(sub_instr, overlap_land, left_land, check_land,  non_distinct, exact_same)
            tmp_reason[start_view+"_"+target_view] = text
        item['reason_text'] =  tmp_reason


with open(reason_path+"train.json", 'w') as f_out:
    json.dump(data, f_out, indent=4)




           


