from models.common import Conv, C3, SPPF, Concat
from models.experimental import attempt_load
from models.yolo import Model, Detect

import torch.nn as nn
import torch
import scipy.cluster.hierarchy as hac
from sklearn.preprocessing import normalize
# from utils.general import non_max_suppression, check_img_size, scale_boxes
# from utils.dataloaders import LoadImages

import numpy as np
import yaml
import copy
import argparse

thres = 0.55

def analyze_cluster(weight_rerange):
    norm1 = normalize(weight_rerange, norm='l1', axis=1)
    z = hac.linkage(norm1, metric='cosine', method='complete')
    ori_labels = hac.fcluster(z, thres, criterion="distance")
    labels_unique = np.unique(ori_labels)
    n_clusters = len(labels_unique)

    # print('original filter : ', weight_rerange.shape[0], 'possible cluster :', n_clusters)

    first_ele = []
    first_ele_label = []
    for idx, label in enumerate(ori_labels):
        if label not in first_ele_label:
            first_ele.append(idx)
            first_ele_label.append(label)

    return first_ele, n_clusters


def prune_conv2d(weight, bias=None):
    weight_rerange = np.reshape(weight, [weight.shape[0], -1])
    first_ele, n_clusters = analyze_cluster(weight_rerange)
    weight_rerange = weight_rerange[first_ele]
    weight_rerange = np.reshape(weight_rerange, [n_clusters, weight.shape[1], weight.shape[2], weight.shape[3]])
    if bias:
        bias = bias[first_ele]

    return weight_rerange, bias, first_ele


def prune_BatchNorm(bn_layer, first_ele, params=None):
    bnorm_weight = bn_layer.weight.data.cpu().numpy()
    params[0] += np.prod(bnorm_weight.shape)
    bnorm_weight = bnorm_weight[first_ele]
    params[1] += np.prod(bnorm_weight.shape)

    bnorm_bias = bn_layer.bias.data.cpu().numpy()
    params[0] += np.prod(bnorm_bias.shape)
    bnorm_bias = bnorm_bias[first_ele]
    params[1] += np.prod(bnorm_bias.shape)

    bn_layer.weight = torch.nn.Parameter(torch.from_numpy(bnorm_weight))
    bn_layer.bias = torch.nn.Parameter(torch.from_numpy(bnorm_bias))
    bn_layer.num_features = bnorm_weight.shape[0]

    bnorm_rm = bn_layer.running_mean.cpu().numpy()
    params[0] += np.prod(bnorm_rm.shape)
    bnorm_rm = bnorm_rm[first_ele]
    params[1] += np.prod(bnorm_rm.shape)

    bnorm_rv = bn_layer.running_var.cpu().numpy()
    params[0] += np.prod(bnorm_rv.shape)
    bnorm_rv = bnorm_rv[first_ele]
    params[1] += np.prod(bnorm_rv.shape)

    bn_layer.running_mean = torch.from_numpy(bnorm_rm)
    bn_layer.running_var = torch.from_numpy(bnorm_rv)


def prune_Conv(layer, first_ele=None, params=None):
    weight = layer.conv.weight.data.cpu().numpy()

    if layer.conv.bias:
        bias = layer.conv.bias.data.cpu().numpy()
        params[0] += np.prod(weight.shape) + bias.shape[0]
    else:
        bias = None
        params[0] += np.prod(weight.shape)

    num_origin_filter = weight.shape[0]

    # rerange weight
    if first_ele is not None:
        weight = np.transpose(weight, (1, 0, 2, 3))
        weight = weight[first_ele]
        weight = np.transpose(weight, (1, 0, 2, 3))

    # prune conv
    pruned_weight, pruned_bias, first_ele = prune_conv2d(weight, bias)

    if layer.conv.bias:
        params[1] += np.prod(pruned_weight.shape) + pruned_bias.shape[0]
    else:
        params[1] += np.prod(pruned_weight.shape)

    layer.conv.out_channels = pruned_weight.shape[0]
    layer.conv.in_channels = pruned_weight.shape[1]

    layer.conv.weight = torch.nn.Parameter(torch.from_numpy(pruned_weight))

    if layer.conv.bias:
        layer.conv.bias = torch.nn.Parameter(torch.from_numpy(pruned_bias))

    layer.first_ele = first_ele
    layer.num_origin_filter = num_origin_filter

    # rerange batch norm
    if hasattr(layer, 'bn'):
        prune_BatchNorm(layer.bn, first_ele, params)


def prune_C3(layer, first_ele=None, params=None):
    # prune cv1
    prune_Conv(layer.cv1, first_ele, params)

    # prune bottle_necks
    btn_first_ele = layer.cv1.first_ele.copy()
    for btn in layer.m:
        # rerange cv1
        weight = btn.cv1.conv.weight.data.cpu().numpy()

        if btn.cv1.conv.bias:
            bias = btn.cv1.conv.bias.data.cpu().numpy()
            params[0] += np.prod(weight.shape) + bias.shape[0]
        else:
            bias = None
            params[0] += np.prod(weight.shape)

        weight = np.transpose(weight, (1, 0, 2, 3))
        weight = weight[btn_first_ele]
        weight = np.transpose(weight, (1, 0, 2, 3))

        pruned_weight = weight
        pruned_bias = bias

        if btn.cv1.conv.bias:
            params[1] += np.prod(pruned_weight.shape) + pruned_bias.shape[0]
        else:
            params[1] += np.prod(pruned_weight.shape)

        btn.cv1.conv.out_channels = pruned_weight.shape[0]
        btn.cv1.conv.in_channels = pruned_weight.shape[1]

        btn.cv1.conv.weight = torch.nn.Parameter(torch.from_numpy(pruned_weight))

        if btn.cv1.conv.bias:
            btn.cv1.conv.bias = torch.nn.Parameter(torch.from_numpy(pruned_bias))

        btn.cv1.first_ele = btn_first_ele

        # prune cv2 with btn_first_ele
        weight = btn.cv2.conv.weight.data.cpu().numpy()
        if btn.cv2.conv.bias:
            bias = btn.cv2.conv.bias.data.cpu().numpy()
            params[0] += np.prod(weight.shape) + bias.shape[0]
        else:
            bias = None
            params[0] += np.prod(weight.shape)

        pruned_weight = weight[btn_first_ele]

        if btn.cv2.conv.bias:
            pruned_bias = bias[btn_first_ele]
            params[1] += np.prod(pruned_weight.shape) + pruned_bias.shape[0]
        else:
            params[1] += np.prod(pruned_weight.shape)

        num_filter_m = pruned_weight.shape[0]

        btn.cv2.conv.out_channels = pruned_weight.shape[0]
        btn.cv2.conv.in_channels = pruned_weight.shape[1]

        btn.cv2.conv.weight = torch.nn.Parameter(torch.from_numpy(pruned_weight))

        if btn.cv2.conv.bias:
            btn.cv2.conv.bias = torch.nn.Parameter(torch.from_numpy(pruned_bias))

        if hasattr(btn.cv2, 'bn'):
            prune_BatchNorm(btn.cv2.bn, btn_first_ele, params)

        btn.cv2.first_ele = btn_first_ele

    layer.m.first_ele = btn_first_ele

    # prune cv2
    prune_Conv(layer.cv2, first_ele, params)

    concat_first_ele = np.concatenate((layer.m.first_ele, [num_filter_m + i for i in layer.cv2.first_ele]))

    # prune cv3
    prune_Conv(layer.cv3, concat_first_ele, params)
    layer.first_ele = layer.cv3.first_ele
    layer.num_origin_filter = layer.cv3.num_origin_filter


def prune_SPPF(layer, first_ele=None, params=None):
    # prune cv1
    prune_Conv(layer.cv1, first_ele, params)

    concat_first_ele = np.concatenate((layer.cv1.first_ele,
                                       [layer.cv1.num_origin_filter + i for i in layer.cv1.first_ele],
                                       [layer.cv1.num_origin_filter * 2 + i for i in layer.cv1.first_ele],
                                       [layer.cv1.num_origin_filter * 3 + i for i in layer.cv1.first_ele]))
    # prune cv2
    prune_Conv(layer.cv2, concat_first_ele, params)
    layer.first_ele = layer.cv2.first_ele
    layer.num_origin_filter = layer.cv2.num_origin_filter


def prune_Upsampling(layer, first_ele, num_origin_filter):
    layer.first_ele = first_ele
    layer.num_origin_filter = num_origin_filter


def prune_Concat(layer, inp_layers):
    concat_first_ele_list = copy.deepcopy(inp_layers[0].first_ele)
    layer.num_origin_filter = copy.deepcopy(inp_layers[0].num_origin_filter)

    for idx, inp_layer in enumerate(inp_layers[1:]):
        concat_first_ele_list.extend([inp_layers[idx - 1].num_origin_filter *
                                     (idx + 1) + i for i in inp_layer.first_ele])
        layer.num_origin_filter += inp_layer.num_origin_filter

    layer.first_ele = concat_first_ele_list


def prune_Detect(layer, first_ele_list, params=[0, 0]):
    for first_ele, detect_module in zip(first_ele_list, layer.m):
        # rerange detect module
        weight, bias = detect_module.weight.data.cpu().numpy(), detect_module.bias.data.cpu().numpy()

        params[0] += np.prod(weight.shape) + bias.shape[0]
        weight = np.transpose(weight, (1, 0, 2, 3))
        weight = weight[first_ele]
        weight = np.transpose(weight, (1, 0, 2, 3))
        params[1] += np.prod(weight.shape) + bias.shape[0]

        detect_module.out_channels = weight.shape[0]
        detect_module.in_channels = weight.shape[1]

        detect_module.weight = torch.nn.Parameter(torch.from_numpy(weight))
        detect_module.first_ele = copy.deepcopy(first_ele)

def prune_model_from_ckpt(path):
    ckpt = torch.load(path)
    model = ckpt['model'].model

    first_ele = None
    graph = []
    params = [0,0]

    arch = ckpt['model'].yaml['backbone'] + ckpt['model'].yaml['head']

    for i,layer in zip(arch,model):
        if isinstance(layer,Conv):
            prune_Conv(layer,first_ele,params)
            first_ele = layer.first_ele
        
        if isinstance(layer,C3):
            prune_C3(layer,first_ele,params)
            first_ele = layer.first_ele
        
        if isinstance(layer,SPPF):
            prune_SPPF(layer,first_ele,params)
            first_ele = layer.first_ele
        
        if isinstance(layer,Concat):
            prune_Concat(layer,[graph[li] for li in i[0]])
            first_ele = layer.first_ele
        
        if isinstance(layer,nn.Upsample):
            prune_Upsampling(layer,graph[-1].first_ele,graph[-1].num_origin_filter)
            first_ele = layer.first_ele
        
        if isinstance(layer,Detect):
            first_ele_list = [graph[idx].first_ele for idx in i[0]]
            prune_Detect(layer,first_ele_list,params)
        
        graph.append(layer)
    print('params before prune : ',params[0])
    print('params after pruned : ',params[1])
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--pruned_weight_path',type=str,default=None)
    parser.add_argument('--thres',type=float,default=0.4)

    args = parser.parse_args()
    global thres
    thres = args.thres
    pruned_model = prune_model_from_ckpt(args.weight_path)

    if args.pruned_weight_path is not None:
        torch.save(pruned_model,args.pruned_weight_path)

if __name__=='__main__':
    main()

