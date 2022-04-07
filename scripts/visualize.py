import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import tqdm

NUM_SENTENCES = 3
MAX_TOKENS = 300
MAX_TOP_PREDECESSORS = 3
NUM_GENERATIONS = 3

def get_top_n_valid_predecessors(graph, node, n):
    predecessors = graph.predecessors(node)
    predec_weights = []
    for predecessor in predecessors:
        num_entities = graph[predecessor][node]['entities']
        if num_entities > 1:
            predec_weights.append((predecessor, num_entities))

    sorted_predecessors = [predec[0] for predec in sorted(predec_weights, key=lambda x: x[1], reverse=True)]

    num_predecessors = graph.in_degree(node)
    return sorted_predecessors[:min(n, num_predecessors)]

def get_n_gen_ancestors(graph, node, num_gens, num_predecessors):
    if num_gens == 0:
        return set([node])

    ancestors = set()
    predecessors = get_top_n_valid_predecessors(graph, node, num_predecessors)
    for predecessor in predecessors:
        sub_ancestors = get_n_gen_ancestors(graph, predecessor, num_gens - 1, num_predecessors)
        ancestors.add(predecessor)
        ancestors.update(sub_ancestors)

    return ancestors

def visualize_graph_context(raw_dir, dataset_name):
    # load graph from file
    nodes_df = pd.read_csv(os.path.join(raw_dir, f"{dataset_name}_entity_dag_nodes.csv"))
    edges_df = pd.read_csv(os.path.join(raw_dir, f"{dataset_name}_entity_dag_edges.csv"))

    # drop nan rows
    nodes_df = nodes_df.dropna()

    # drop duplicate titles
    nodes_df = nodes_df.drop_duplicates(subset='title')

    graph = nx.DiGraph()

    node_mapping = {}
    # add nodes from dataframe
    print('Loading nodes into graph')
    for idx, node_row in tqdm.tqdm(nodes_df.iterrows(), total=len(nodes_df)):
        node_mapping[node_row['id']] = idx

        node_text = node_row['title']

        graph.add_node(idx, node_text=node_text)

    # add edges from dataframe
    print('Loading edges into graph')
    for idx, edge_row in tqdm.tqdm(edges_df.iterrows(), total=len(edges_df)):
        if edge_row['old_id'] not in node_mapping or edge_row['new_id'] not in node_mapping:
            continue

        old_id = node_mapping[edge_row['old_id']]
        new_id = node_mapping[edge_row['new_id']]

        if graph.has_edge(old_id, new_id):
            graph[old_id][new_id]['entities'] += 1
        else:
            graph.add_edge(old_id, new_id, entities=1)

    # process network into torch compat shape
    print('Create context/target pairs from graph')
    for target_node, tokens in tqdm.tqdm(graph.nodes(data=True), total=graph.number_of_nodes()):
        if graph.in_degree(target_node) == 0:
            continue

        predecessors = get_top_n_valid_predecessors(graph, target_node, 1)
        if len(predecessors) == 0:
            continue

        context_node = predecessors[0]

        ancestors = get_n_gen_ancestors(graph, target_node, NUM_GENERATIONS, MAX_TOP_PREDECESSORS)
        ancestors.add(target_node)
        graph_context = nx.induced_subgraph(graph, ancestors).copy()

        pos = nx.nx_agraph.graphviz_layout(graph_context)

        node_colours = []
        for node in graph_context.nodes():
            if node == target_node:
                node_colours.append('red')
            elif node == context_node:
                node_colours.append('blue')
            else:
                node_colours.append('pink')

        avg_x_pos = sum(v[0] for k,v in pos.items()) / len(pos)

        nx.draw_networkx_nodes(
            graph_context, pos, linewidths=1,
            node_size=500, node_color=node_colours, alpha=0.9
        )
        nx.draw_networkx_edges(
            graph_context, pos, edge_color='black', width=1, alpha=0.9,
        )
        nx.draw_networkx_labels(
            graph_context, pos = {k:([v[0] + 0.4*(avg_x_pos - v[0]), v[1]]) for k,v in pos.items()}, alpha=0.9,
            labels={node: graph_context.nodes[node]['node_text'] for node in graph_context.nodes()}
        )
        nx.draw_networkx_edge_labels(
            graph_context, pos,
            edge_labels={edge: graph_context.edges[edge]['entities'] for edge in graph_context.edges()},
            font_color='red'
        )
        
        plt.show()

def main(args):
    if args.visualize == 'context':
        visualize_graph_context(args.dataset_dir, args.dataset_name)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--visualize')
    arg_parser.add_argument('--dataset-dir')
    arg_parser.add_argument('--dataset-name')
    args = arg_parser.parse_args()
    main(args)