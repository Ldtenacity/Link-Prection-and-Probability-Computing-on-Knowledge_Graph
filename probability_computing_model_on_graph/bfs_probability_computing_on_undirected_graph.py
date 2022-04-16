import json
import collections
import heapq

edges = {}
nodes = {}
invnodes = {}
dtlb = {}
dtpp = {}

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


def compute_the_path(n: int, edges: list[list], succProb: list[float], start: int, end: int) -> float:
    if not edges or not edges[0]:
        return 0

    st_maps = collections.defaultdict(list)
    for i, (s, e) in enumerate(edges):
        st_maps[s].append((e, succProb[i]))
        st_maps[e].append((s, succProb[i]))

    ans = 0
    queue = collections.deque([(start, 1)])
    visited = {start: 0}
    while queue:
        cur_node, cur_prob = queue.popleft()
        for next_node, p in st_maps[cur_node]:
            next_prob = cur_prob * p
            if next_node == end:
                ans = max(ans, next_prob)
                continue

            if next_prob > ans and (next_node not in visited or visited[next_node] < next_prob):
                visited[next_node] = next_prob
                queue.append((next_node, next_prob))
    return ans


puredges = []
weights = []
for key in edges.keys():
    puredges.append([invnodes[key], invnodes[edges[key][0]]])
    weights.append(edges[key][-1])

print("BFS algorithm for probability computing in undirected graph")
print("Please input start point: ")
start = eval(input())
print("Please input end point: ")
end = eval(input())
print("Computing...")
val = compute_the_path(len(nodes), puredges, weights, invnodes[start], invnodes[end])
print("The probability of traversing from Node {} to Node {} is {}".format(start, end, val))