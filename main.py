import csv
from datetime import datetime

import networkx as nx
import pm4py
from graphviz import Graph
from collections import defaultdict

from matplotlib import pyplot as plt

from totemAlgorithm import mine_totem

def main():
    # sind in order-management.json die oids in ocel.object_changes mit types verwechselt worden?
    #ocel = pm4py.read_ocel2('./data/order-management.json')
    #ocel = pm4py.read_ocel('./data/running-example.jsonocel')
    ocel = pm4py.read_ocel2('./data/ocel2-p2p.json')

    #print(ocel.objects[ocel.objects['ocel:type'] == 'products'])
    flatten_save_ocel(ocel)
    return
    #lc_data = mine_totem(ocel, 0.9)

    lc_data = {
        "Container": {
            "Transport Document": ("1..*", "1..*"),
            "Vehicle": ("1..*", "1")
        },
        "Vehicle": {
            "Transport Document": ("1..*", "1..*")
        },
        "Transport Document": {
            "Customer Order": ("1..*", "1")
        }
    }

    #print(first_occurrence(ocel))
    print(ocel.object_changes)
    print(ocel.object_changes.dtypes)

    generate_uml_diagram(lc_data)

    generate_lifecycles(ocel)

    return
    #print(ocel.events[ocel.events['ocel:eid'] == 'place_o-990001'])
    #print(ocel.relations.dtypes)
    #print(ocel.relations[ocel.relations['ocel:eid'] == 'place_o-990001'][['ocel:eid', 'ocel:activity','ocel:oid', 'ocel:type']])
    #ocpn = pm4py.discover_oc_petri_net(ocel)
    #pm4py.save_vis_ocpn(ocpn, './data/ocpn.png')
    print(ocel.events['ocel:activity'].values.unique())
    print('------------- events --------------')
    print(ocel.events.dtypes)
    print('------------- relations -----------')
    print(ocel.relations.dtypes)
    print('------------- objects -------------')
    print(ocel.objects.dtypes)
    print('------------- o2o -----------------')
    print(ocel.o2o.dtypes)


    #todo: how can I represent "new" state?/ is new always the first state?
    #if first object occurrence is input in activity -> should be "new", if it is output -> should be activity name
    #what happens when activity consumes the object and doesn't output it?
    object_lifecycles : dict[str, set[str]] = dict()

    for index, row in ocel.relations.iterrows():
        if row['ocel:type'] not in object_lifecycles:
            object_lifecycles[row['ocel:type']] = set()

        object_lifecycles[row['ocel:type']].add(row['ocel:activity'])

    for key, entry in object_lifecycles.items():
        print(key, ": ", entry)



def generate_uml_diagram(lc_data: dict[str, dict[str, tuple[str, str]]], output_filename="generated_data/uml_diagram"):
    dot = Graph(format="png", engine="dot")
    #dot.attr(splines="line", ranksep="1.0", nodesep="0.8")
    #todo zählen
    dot.attr(splines="ortho", ranksep="1.5", nodesep="1.2", concentrate="true", mindist="0.5")
    dot.attr(rankdir="TB")

    # Collect all unique classes (both parents & children)
    all_classes = set(lc_data.keys())
    for connections in lc_data.values():
        all_classes.update(connections.keys())

    # Add UML classes (all forced to be boxes)
    for class_name in all_classes:
        dot.node(class_name, shape="box", style="filled", fillcolor="white")

    # Add relationships with multiplicities at the ends
    for parent_class, connections in lc_data.items():
        for child_class, (multiplicity_parent, multiplicity_child) in connections.items():
            dot.edge(parent_class, child_class, arrowhead="none", taillabel=f" {multiplicity_parent} ", headlabel=f" {multiplicity_child} ")

    # Save UML diagram as PNG
    dot.render(output_filename, cleanup=True)
    print(f"UML Diagram saved as {output_filename}.png")



def wrap_labels(label):
    words = label.split()

    if len(words) == 1:
        return label

    mid = len(words) // 2  # Find the middle word
    return "\n".join([" ".join(words[:mid]), " ".join(words[mid:])])



def flatten_save_ocel(ocel):
    ocel_object_types = ocel.objects[ocel.object_type_column].values.unique()

    for object_type in ocel_object_types:
        flattened_ocel = pm4py.ocel_flattening(ocel, object_type)
        bpmn = pm4py.discover_bpmn_inductive(flattened_ocel)

        # removes start and end nodes
        end_nodes = set(filter(filter_end, bpmn.get_nodes()))
        for end_node in end_nodes:
            bpmn.remove_node(end_node)
        start_nodes = set(filter(filter_start, bpmn.get_nodes()))
        for start_node in start_nodes:
            bpmn.remove_node(start_node)


        pm4py.save_vis_bpmn(bpmn, f"generated_data/bpmn_{object_type}.png")


def filter_end(node):
    if str(node).__contains__('@end'):
        return True
    return False

def filter_start(node):
    if str(node).__contains__('@start'):
        return True
    return False

# find the timestamp for the first occurrence of each oid
def first_occurrence(ocel):
    first_object_occurrence : dict[str, datetime] = dict()

    for index, row in ocel.object_changes.iterrows():
        if row['ocel:oid'] not in first_object_occurrence:
            first_object_occurrence[row['ocel:oid']] = row['ocel:timestamp'].tz_localize(None)
            continue

        if first_object_occurrence[row['ocel:oid']] > row['ocel:timestamp'].tz_localize(None):
            first_object_occurrence[row['ocel:oid']] = row['ocel:timestamp'].tz_localize(None)

    return first_object_occurrence



def get_sorted_oid_paths(ocel):
    object_paths_new : dict[str, list[tuple[str, datetime]]] = dict()
    object_paths : dict[str, list[str]] = dict()

    first_occurrences = first_occurrence(ocel)

    for index, row in ocel.relations.iterrows():
        if row['ocel:oid'] not in object_paths_new:
            object_paths_new[row['ocel:oid']] = [(row['ocel:activity'], row['ocel:timestamp'].tz_localize(None))]
            continue
        object_paths_new[row['ocel:oid']].append((row['ocel:activity'], row['ocel:timestamp'].tz_localize(None)))

    for key in object_paths_new:
        object_paths_new[key].sort(key=lambda x: x[1])
        # todo implement input or output object probably here, maybe add try catch
        if key in first_occurrences.keys():
            if first_occurrences[key] < object_paths_new[key][0][1]:
                print("in")
                object_paths_new[key].insert(0, ('init', first_occurrences[key]))

        object_paths[key] = [path for path, _ in object_paths_new[key]]

    return object_paths



def build_graph(paths):
    graph = defaultdict(set)
    for path in paths:
        for i in range(len(path) - 1):
            graph[path[i]].add(path[i + 1])
    return {node: list(neighbors) for node, neighbors in graph.items()}

def create_nx_graph(graph_dict):
    G = nx.DiGraph()
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

def draw_and_save_graph(G, filename):
    plt.figure(figsize=(8, 6))
    #pos = nx.spring_layout(G)  # Position nodes for visualization
    pos = nx.kamada_kawai_layout(G)

    wrapped_labels = {node: wrap_labels(node) for node in G.nodes()}

    # Draw the graph with better spacing and visibility
    nx.draw(G, pos, labels= wrapped_labels, node_color='white', edge_color='black',
            node_size=2500, font_size=10, edgecolors='black', linewidths=1)

    plt.savefig(filename, dpi=300)  # Higher DPI for better quality
    plt.close()


def generate_lifecycles(ocel):
    object_id_to_type : dict[str, str] = dict()
    object_paths = get_sorted_oid_paths(ocel)

    for index, row in ocel.objects.iterrows():
        object_id_to_type[row['ocel:oid']] = row['ocel:type']

    type_to_paths = defaultdict(list)
    for obj_id, path in object_paths.items():
        obj_type = object_id_to_type[obj_id]
        type_to_paths[obj_type].append(path)

    type_to_graph = {obj_type: build_graph(paths) for obj_type, paths in type_to_paths.items()}
    nx_graphs = {obj_type: create_nx_graph(graph) for obj_type, graph in type_to_graph.items()}

    for obj_type, G in nx_graphs.items():
        draw_and_save_graph(G, f"generated_data/graph_{obj_type}.png")

    with open("generated_data/graph_edges.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ObjectType", "Source", "Target"])

        for obj_type, G in nx_graphs.items():
            for edge in G.edges():
                writer.writerow([obj_type, edge[0], edge[1]])




if __name__ == '__main__':
    main()