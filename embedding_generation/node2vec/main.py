# -*- coding:utf-8 -*-
from args import args_parser
from node2vec import node2vec
import networkx as nx
import numpy as np
import json
import collections

edges = {}
nodes = {}
invnodes = {}
dtlb = {}
dtpp = {}

def main():
    args = args_parser()

    G = nx.Graph()  # 无多重边有向图

    with open('links.json', 'r', encoding="utf-8") as file:
        str = file.read()
        data = json.loads(str)
        for i in range(0, len(data)):
            sid = data[i]["p"]["segments"][0]["relationship"]["start"]
            eid = data[i]["p"]["segments"][0]["relationship"]["end"]
            name = data[i]["p"]["segments"][0]["start"]["properties"]["name"]
            label = data[i]["p"]["segments"][0]["start"]["labels"][0]
            weight = data[i]["p"]["segments"][0]["relationship"]["properties"]["weight"]
            edges[sid] = [eid, name, label, weight]
            edges[eid] = [sid, name, label, weight]

    with open('nodes.json', 'r', encoding="utf-8") as file:
        str = file.read()
        data = json.loads(str)
        for i in range(0, len(data)):
            nodes[i] = data[i]["n"]["identity"]
            invnodes[data[i]["n"]["identity"]] = i
            dtlb[data[i]["n"]["identity"]] = data[i]["n"]["labels"][0]
            dtpp[data[i]["n"]["properties"]["name"]] = data[i]["n"]["properties"]["name"]

    puredges = [] # --> edges [a->b]
    weights = [] # --> weights
    for key in edges.keys():
        puredges.append([invnodes[key], invnodes[edges[key][0]]])
        weights.append(edges[key][-1])

    for i in range(0,len(nodes)):
        G.add_node(i)  # 添加一个节点
    for i in range(0,len(puredges)):
        G.add_edge(puredges[i][0], puredges[i][1], weight=weights[i])  # 添加一条边


    print("nodes: ", G.nodes())  # 输出所有的节点
    print("edges: ", G.edges())  # 输出所有的边
    print("number_of_edges: ", G.number_of_edges())  # 边的条数，只有一条边，就是（2，3）
    print("degree: ", G.degree)  # 返回节点的度
    print("degree_histogram: ", nx.degree_histogram(G))  # 返回所有节点的分布序列4

    vec = node2vec(args, G)
    embeddings = vec.learning_features()
    print(embeddings)
    print(np.array(embeddings).shape)


if __name__ == '__main__':
    main()
