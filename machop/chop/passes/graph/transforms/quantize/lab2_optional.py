import logging
import os
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)

""" OP_LIST = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "sub",
    "batch_norm1d"
) """

OP_LIST = (
    "linear",
    "batch_norm1d"
)

def linear_calc(node_args,has_bias):
    fm = 0
    fa = 0
    bo = 0
    for arg_name, arg_val in node_args.items() : 
        if arg_name == "data_in_0":
            if arg_val["type"] != 'float':
                din_bits = arg_val["precision"][0]
        if arg_name == "weight":
            if arg_val["type"] == 'float':
                fm += arg_val["shape"][0]*arg_val["shape"][1]
            else:
                bo += arg_val["shape"][0]*arg_val["shape"][1]*arg_val["precision"][0]*din_bits

        if has_bias == 1:
            if arg_name == "bias":
                if arg_val["type"] == 'float':
                    fa += arg_val["shape"][0]

    return  {"f_mul":fm,"f_add":fa,"bit_op":bo}  


def batch_norm1d_calc(node_args):  #batch_norm1d is not a quantizable op yet. Integer calculation has just been added for future integer implementations
    fm = 0
    fa = 0
    bo = 0
    for arg_name, arg_val in node_args.items() : 
        if arg_name == "data_in_0":
            if arg_val["type"] != 'float':
                din_bits = arg_val["precision"][0]
        if arg_name == "weight":
            if arg_val["type"] == 'float':
                fm += arg_val["shape"][0]
            else:
                bo += arg_val["shape"][0]*arg_val["precision"][0]*din_bits

       
        if arg_name == "bias":
            if arg_val["type"] == 'float':
                fa += arg_val["shape"][0]

    return  {"f_mul":fm,"f_add":fa,"bit_op":bo}      

def report_flops_bitops_analysis_pass(graph):
    rows = []
    res = {"f_mul":0,"f_add":0,"bit_op":0}
    for node in graph.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        node_args = node.meta["mase"].parameters["common"]["args"]
        if mase_op not in OP_LIST:
            continue
        else:
            if mase_op == "linear":
                has_bias = len(node.all_input_nodes) > 2
                res_node = linear_calc(node_args,has_bias)
            elif mase_op == "batch_norm1d":
                res_node = batch_norm1d_calc(node_args)   

        for key in res:
            if key in res_node:
                res[key] += res_node[key]  

    return res