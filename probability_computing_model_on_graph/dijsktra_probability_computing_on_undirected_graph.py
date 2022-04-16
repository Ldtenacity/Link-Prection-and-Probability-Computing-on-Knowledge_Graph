import json
import collections
import heapq
edges = {}
nodes = {}
invnodes = {}
dtlb = {}
dtpp = {}

with open('links.json','r',encoding="utf-8") as file:
    str = file.read()
    data = json.loads(str)
    for i in range(0,len(data)):
        sid = data[i]["p"]["segments"][0]["relationship"]["start"]
        eid = data[i]["p"]["segments"][0]["relationship"]["end"]
        name = data[i]["p"]["segments"][0]["start"]["properties"]["name"]
        label = data[i]["p"]["segments"][0]["start"]["labels"][0]
        weight = data[i]["p"]["segments"][0]["relationship"]["properties"]["weight"]
        edges[sid] = [eid,name,label,weight]
        edges[eid] = [sid,name,label,weight]

with open('nodes.json','r',encoding="utf-8") as file:
    str = file.read()
    data = json.loads(str)
    for i in range(0,len(data)):
        nodes[i] = data[i]["n"]["identity"]
        invnodes[data[i]["n"]["identity"]] = i
        dtlb[data[i]["n"]["identity"]] = data[i]["n"]["labels"][0]
        dtpp[data[i]["n"]["properties"]["name"]] = data[i]["n"]["properties"]["name"]

def compute_the_path(n: int, edges:list[list], succProb: list[float], start: int, end: int) -> float:
    graph = collections.defaultdict(list)
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    edge2prob = {}
    for i, edge in enumerate(edges):
        edge2prob[(edge[0], edge[1])] = succProb[i]
        edge2prob[(edge[1], edge[0])] = succProb[i]
    q = [(-1, start)]
    prob_vec = [0] * n
    while q:
        prob, node = heapq.heappop(q)
        if node == end:
            return -prob
        for nei in graph[node]:
            new_prob = edge2prob[(nei, node)] * (-prob)
            if new_prob > prob_vec[nei]:
                heapq.heappush(q, (-new_prob, nei))
                prob_vec[nei] = new_prob
    return prob_vec[end]

puredges = []
weights = []
for key in edges.keys():
    puredges.append([invnodes[key],invnodes[edges[key][0]]])
    weights.append(edges[key][-1])

print("Dijsktra algorithm for probability computing in undirected graph")
print("Please input start point: ")
start = eval(input())
print("Please input end point: ")
end = eval(input())
print("Computing...")
val = compute_the_path(len(nodes), puredges, weights, invnodes[start], invnodes[end])
print("The probability of traversing from Node {} to Node {} is {}".format(start,end,val))