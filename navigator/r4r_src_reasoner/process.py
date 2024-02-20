
from hashlib import new
import json
import networkx as nx
import os

from utils import load_nav_graphs
from collections import defaultdict

input_path = "/localscratch/zhan1624/VLN-interactive/interactive_setting/output/annotation.txt"
total_gt = {}
distances = {}
all_scenery_list = os.listdir("/egr/research-hlr/joslin/Matterdata/v1/scans/")
scans = [i for i in all_scenery_list]
graphs = load_nav_graphs(scans)
distances = {}
for scan,G in graphs.items(): # compute all shortest paths
    distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

with open("/localscratch/zhan1624/VLN-interactive/data/R2R_val_seen.json") as f_in:
    gt_data = json.load(f_in)

for each_gt in gt_data:
    total_gt[str(each_gt['path_id'])] = {} 
    total_gt[str(each_gt['path_id'])]['path'] = each_gt['path']
    total_gt[str(each_gt['path_id'])]['scan'] = each_gt['scan']

def get_nearest(scan, goal_id, path):
        near_id = path[0]
        near_d = distances[scan][near_id][goal_id]
        for item in path:
            if "," in item:
                item = item[:-1]
            d = distances[scan][item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

def score_item(instr_id, path):
    ''' Calculate error based on the final position in trajectory, and also
        the closest position (oracle stopping rule).
        The path contains [view_id, angle, vofv] '''
    scores = defaultdict(list)
    gt = total_gt[instr_id.split('_')[-2]]
    start = gt['path'][0]
    assert start == path[0], 'Result trajectories should include the start position'
    goal = gt['path'][-1]
    final_position = path[-1]  # the first of [view_id, angle, vofv]
    nearest_position = get_nearest(gt['scan'], goal, path)
    scores['nav_errors'].append(distances[gt['scan']][final_position][goal])
    scores['oracle_errors'].append(distances[gt['scan']][nearest_position][goal])
    scores['trajectory_steps'].append(len(path)-1)
    distance = 0  # length of the path in meters
    prev = path[0]
    for curr in path[1:]:
        if "," in curr:
            curr = curr[:-1]
        distance += distances[gt['scan']][prev][curr]
        prev = curr
    scores['trajectory_lengths'].append(distance)
    scores['shortest_lengths'].append(
        distances[gt['scan']][start][goal]
        )
    return scores

with open(input_path) as f_in:
    data = f_in.read()

data = data.split("\n\n")
new_data = []
nav_errors = []
for each_data in data:
    tmp_dict = {}
    each_data = each_data.split(';')
    if each_data[0] == "":
        continue
    instr_id = each_data[0]
    path = each_data[1].split("\n")
    path = [i.strip(',') for i in path if "," in i]
    tmp_dict["instr_id"] = instr_id
    tmp_dict['trajectory'] = path
    new_data.append(tmp_dict)
    # tmp_dict["visible"] = each_data[2]
    # tmp_dict["non-distinctive"] = each_data[3]
    # each_score = score_item(instr_id, path)
    # nav_errors.append(each_score['nav_errors'][0])

# num_successes = len([i for i in nav_errors if i < 3.0])
# success_rate = float(num_successes)/float(len(nav_errors))

with open("/localscratch/zhan1624/VLN-interactive/interactive_setting/output/annotation.json", 'w') as f_out:
    json.dump(new_data, f_out, indent=4)
print('yue')













data.close()
