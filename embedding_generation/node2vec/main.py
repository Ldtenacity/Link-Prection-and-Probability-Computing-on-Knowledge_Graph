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
    '''
    G = nx.davis_southern_women_graph()
    # G = nx.karate_club_graph()
    nodes = list(G.nodes.data())
    # G = nx.karate_club_graph()
    G = nx.les_miserables_graph()
    # for u, v in G.edges:
    #     G.add_edge(u, v, weight=1)
    '''
    '''
    G = nx.Graph()  # 无多重边有向图
    G.clear()  # 清空图

    G = nx.Graph()  # 无多重边有向图
    G.add_nodes_from([0,1])  # 添加多个节点
    G.add_node(2)  # 添加一个节点
    G.add_nodes_from([3, 4, 5, 6])  # 添加多个节点
    #G.add_cycle([1, 2, 3, 4])  # 添加环
    nx.add_cycle(G, [1, 2, 3, 4],weight = 1)
    #G.add_edge(1, 3,weight = 1)  # 添加一条边
    G.add_edges_from([(3, 5 ,{'weight':1}), (3, 6 ,{'weight':1}), (6, 7 ,{'weight':1})])  # 添加多条边
    print(G[3][5])
    #print(G[2][3])
    #G.add_edges_from([(3, 5), (3, 6), (6, 7)],weight = 1)
    #G.remove_node(8)  # 删除一个节点
    #G.remove_nodes_from([9, 10, 11, 12])  # 删除多个节点
    '''

    G = nx.Graph()  # 无多重边有向图

    with open('links.json', 'r', encoding="utf-8") as file:
        str = file.read()
        data = json.loads(str)
        '''
            print(data[0]["p"]["segments"][0]["relationship"]["start"]) # start id: 65
            print(data[0]["p"]["segments"][0]["relationship"]["end"]) # end id: 0
            print(data[0]["p"]["segments"][0]["start"]["labels"][0]) # labels: string Place dt: id->labels
            print(data[0]["p"]["segments"][0]["start"]["properties"]["name"]) # name: string 邢台市第二医院 dt: id->name
            print(data[0]["p"]["segments"][0]["relationship"]["properties"]["weight"]) # weight: 0.4
        '''
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
            # print(data[i]["n"]["identity"])  # identity: int 0 index dt to record all nodes
            nodes[i] = data[i]["n"]["identity"]
            invnodes[data[i]["n"]["identity"]] = i
            # print(data[i]["n"]["labels"][0])  # labels: string Person dt: id->label
            dtlb[data[i]["n"]["identity"]] = data[i]["n"]["labels"][0]
            # print(data[i]["n"]["properties"]["name"])  # name: string 程新花 dt: id->name
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
