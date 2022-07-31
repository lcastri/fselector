from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from netgraph import Graph


SOURCE = 'source'
SCORE = 'score'
PVAL = 'pval'
LAG = 'lag'


def tsdag3(result,
           alpha = None,
           min_width = 1,
           max_width = 5,
           node_color = 'orange',
           edge_color = 'grey',
           font_size = 12,
           save_name = None):
    
    #FIXME: allow empty graph
    if alpha is not None: result['graph'] = __apply_alpha(result, alpha)
    res = __PCMCIres_converter(result)
    
    TAU = __get_max_tau(result)
    layers = TAU + 1
    
    # node definition
    G = nx.DiGraph()
    for l in range(layers):
        for v in res.keys():
            G.add_node(v + "_" + str(TAU - l), layer = l)
            
    # node label definition
    labeldict = {}
    for n in G.nodes():
        if n[-1] == str(TAU):
            labeldict[n] = n[:-2]
            
    # edge definition
    edges = list()
    for l in range(layers - 1):
        for v in res.keys():
            for s in res[v]:
                s_node = s[SOURCE] + "_" + str(s[LAG] + l)
                t_node = v + "_" + str(l)
                edges.append((s_node, t_node, __scale(s[SCORE], min_width, max_width)))
    G.add_weighted_edges_from(edges)
    
    edge_width = list(nx.get_edge_attributes(G, 'weight').values())          

    # plot graph
    pos = nx.multipartite_layout(G, subset_key="layer",)
    fig, ax = plt.subplots(figsize=(8,6))
    
    # draw nodes
    nx.draw_networkx_nodes(G, 
                           pos,
                        #    node_size = 750,
                           node_color = node_color,)
    # draw node label
    nx.draw_networkx_labels(G, 
                            pos = pos,
                            labels = labeldict,
                            font_size = font_size) 
    
    # draw edges
    arrow = nx.draw_networkx_edges(G, 
                                   pos, 
                                   arrows = True,
                                   width = edge_width,
                                   alpha = 1,
                                   edge_color = edge_color,
                                   connectionstyle = "arc3, rad=0.2",
                                   arrowstyle = "-|>",)
    for a, w in zip(arrow, edge_width):
        a.set_mutation_scale(7.5 + w)
        a.set_joinstyle('miter')
        a.set_capstyle('butt')
    
    # time line text drawing
    pos_tau = list(set([pos[p][0] for p in pos]))
    dict_pos_tau = {pos_tau[p] : (len(pos_tau) - 1 - p) for p in range(len(pos_tau))}
    max_y = max([pos[p][1] for p in pos])
    for p in pos_tau:
        if dict_pos_tau[p] == 0:
            ax.text(p, max_y + .3, r"$t$", horizontalalignment='center', fontsize=font_size)
        else:
            ax.text(p, max_y + .3, r"$t-" + str(dict_pos_tau[p]) + "$", horizontalalignment='center', fontsize=font_size)
    
    if save_name is not None:
        plt.savefig(save_name, dpi = 300)
    else:
        plt.show()



def dag2(result,
         alpha = None,
         min_width = 1,
         max_width = 5,
         node_color = 'orange',
         edge_color = 'grey',
         font_size = 12,
         show_edge_labels = False,
         save_name = None):
    
    #FIXME: allow empty graph
    if alpha is not None: result['graph'] = __apply_alpha(result, alpha)
    res = __PCMCIres_converter(result)

    G = nx.DiGraph()

    # add nodes
    G.add_nodes_from(res.keys())
    border = {t: 0 for t in res.keys()}
    for t in res.keys():
        for s in res[t]:
            if t == s[SOURCE]: border[t] =  __scale(s[SCORE], min_width, max_width)
    node_label = {t: s[LAG] for t in res.keys() for s in res[t] if t == s[SOURCE]}

    # edges definition
    edges = [(s[SOURCE], t, __scale(s[SCORE], min_width, max_width)) for t in res.keys() for s in res[t] if t != s[SOURCE]]
    G.add_weighted_edges_from(edges)
    edge_width = list(nx.get_edge_attributes(G, 'weight').values())          
    edge_label = {(s[SOURCE], t): s[LAG] for t in res.keys() for s in res[t] if t != s[SOURCE]}

    plt.figure(figsize=(6, 4))
    pos = nx.circular_layout(G)
    
    # draw nodes
    nodes = nx.draw_networkx_nodes(G, 
                                    pos,
                                    # node_size = 800,
                                    node_color = node_color,
                                    linewidths = list(border.values()),
                                    edgecolors = edge_color,)
    nodes.set_zorder(1)
    
    # nx.draw(G, pos, with_labels = True,font_size = font_size, arrows = True,
    #                                width = edge_width,
    #                                alpha = 1,
    #                                edge_color = edge_color,
    #                                connectionstyle = "arc3, rad=0.2",
    #                                arrowstyle = "->, head_width=0.4, head_length=1", )

    
    # draw node label
    nx.draw_networkx_labels(G, 
                            pos = pos,
                            font_size = font_size) 
    
    # draw edges
    arrow = nx.draw_networkx_edges(G, 
                                   pos, 
                                   arrows = True,
                                   width = edge_width,
                                   alpha = 1,
                                   edge_color = edge_color,
                                   connectionstyle = "arc3, rad=0.2",
                                   arrowstyle = "->, head_width=0.4, head_length=1",)
    for a, w in zip(arrow, edge_width):
        a.set_zorder = 2
        a.set_mutation_scale(w)
        a.set_joinstyle('miter')
        a.set_capstyle('butt')


    if show_edge_labels:
        nx.draw_networkx_edge_labels(G, 
                                     pos,
                                     edge_labels = edge_label,
                                     font_color='k',
                                     font_size = font_size,
                                     label_pos = 0.65
                                    )
    if save_name is not None:
        plt.savefig(save_name, dpi = 300)
    else:
        plt.show()


def dag(result,
        alpha = None,
        min_width = 1,
        max_width = 5,
        node_color = 'orange',
        edge_color = 'grey',
        font_size = 12,
        show_edge_labels = True,
        save_name = None):
    """
    build a dag

    Args:
        result (dict): result from pcmci
        alpha (float): significance level. Defaults to None
        min_width (int, optional): minimum linewidth. Defaults to 1.
        max_width (int, optional): maximum linewidth. Defaults to 5.
        node_color (str, optional): node color. Defaults to 'orange'.
        edge_color (str, optional): edge color. Defaults to 'grey'.
        font_size (int, optional): font size. Defaults to 12.
        show_edge_labels (bool, optional): bit to show the time-lag label of the dependency on the edge. Defaults to True.
        save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
    """

    #FIXME: allow empty graph
    if alpha is not None: result['graph'] = __apply_alpha(result, alpha)
    res = __PCMCIres_converter(result)

    G = nx.DiGraph()

    # add nodes
    G.add_nodes_from(res.keys())
    
    border = dict()
    for t in res.keys():
        border[t] = 0
        for s in res[t]:
            if t == s[SOURCE]:
                border[t] = __scale(s[SCORE], min_width, max_width)
    
    if show_edge_labels:
        node_label = {t: s[LAG] for t in res.keys() for s in res[t] if t == s[SOURCE]}
    else:
        node_label = None

    # edges definition
    edges = [(s[SOURCE], t) for t in res.keys() for s in res[t] if t != s[SOURCE]]
    G.add_edges_from(edges)
    
    edge_width = {(s[SOURCE], t): __scale(s[SCORE], min_width, max_width) for t in res.keys() for s in res[t] if t != s[SOURCE]}
    if show_edge_labels:
        edge_label = {(s[SOURCE], t): s[LAG] for t in res.keys() for s in res[t] if t != s[SOURCE]}
    else:
        edge_label = None

    fig, ax = plt.subplots(figsize=(8,6))

    if edges:
        a = Graph(G, 
                node_layout = 'dot',
                node_size = 8,
                node_color = node_color,
                node_labels = node_label,
                node_edge_width = border,
                node_label_fontdict = dict(size=font_size),
                node_edge_color = edge_color,
                node_label_offset = 0.15,
                node_alpha = 1,
                
                arrows = True,
                edge_layout = 'curved',
                edge_label = show_edge_labels,
                edge_labels = edge_label,
                edge_label_fontdict = dict(size=font_size),
                edge_color = edge_color, 
                edge_width = edge_width,
                edge_alpha = 1,
                edge_zorder = 1,
                edge_label_position = 0.35)
        
        nx.draw_networkx_labels(G, 
                                pos = a.node_positions,
                                labels = {n: n for n in G},
                                font_size = font_size)

    if save_name is not None:
        plt.savefig(save_name, dpi = 300)
    else:
        plt.show()
        
        
def __get_max_tau(result):
    """
    calculate tau from the pcmci result

    Args:
        result (dict): pcmci result

    Returns:
        int: max tau lag
    """
    return result['graph'].shape[2] - 1
       
        
def ts_dag(result,
           alpha = None,
           min_width = 1,
           max_width = 5,
           node_color = 'orange',
           edge_color = 'grey',
           font_size = 12,
           save_name = None):
    """
    build a timeseries dag

    Args:
        result (dict): result from pcmci
        alpha (float): significance level. Defaults to None
        min_width (int, optional): minimum linewidth. Defaults to 1.
        max_width (int, optional): maximum linewidth. Defaults to 5.
        node_color (str, optional): node color. Defaults to 'orange'.
        edge_color (str, optional): edge color. Defaults to 'grey'.
        font_size (int, optional): font size. Defaults to 12.
        save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.

    """
    #FIXME: allow empty graph
    if alpha is not None: result['graph'] = __apply_alpha(result, alpha)
    res = __PCMCIres_converter(result)
    # add nodes
    TAU = __get_max_tau(result)
    G = nx.grid_2d_graph(TAU + 1, len(res.keys()))
    pos = dict()
    for n in G.nodes():
        if n[0] == 0:
            pos[n] = (n[0], n[1]/2)
        else:
            pos[n] = (n[0] + .5, n[1]/2)
    scale = max(pos.values())
    G.remove_edges_from(G.edges())

    # edges definition
    edges = list()
    edge_width = dict()
    for t in res.keys():
        for s in res[t]:
            s_index = len(res.keys())-1 - list(res.keys()).index(s[SOURCE])
            t_index = len(res.keys())-1 - list(res.keys()).index(t)
            s_node = (TAU - s[LAG], s_index)
            t_node = (TAU, t_index)
            edges.append((s_node, t_node))
            edge_width[(s_node, t_node)] = __scale(s[SCORE], min_width, max_width)
    G.add_edges_from(edges)

    # label definition
    labeldict = {}
    for n in G.nodes():
        if n[0] == 0:
            labeldict[n] = list(res.keys())[len(res.keys()) - 1 - n[1]]

    fig, ax = plt.subplots(figsize=(8,6))

    # time line text drawing
    pos_tau = set([pos[p][0] for p in pos])
    max_y = max([pos[p][1] for p in pos])
    for p in pos_tau:
        if abs(int(p) - TAU) == 0:
            ax.text(p, max_y + .3, r"$t$", horizontalalignment='center', fontsize=font_size)
        else:
            ax.text(p, max_y + .3, r"$t-" + str(abs(int(p) - TAU)) + "$", horizontalalignment='center', fontsize=font_size)

    Graph(G,
          node_layout = {p : np.array(pos[p]) for p in pos},
          node_size = 10,
          node_color = node_color,
          node_labels = labeldict,
          node_label_offset = 0,
          node_edge_width = 0,
          node_label_fontdict = dict(size=font_size),
          node_alpha = 1,
          
          arrows = True,
          edge_layout = 'curved',
          edge_label = False,
          edge_color = edge_color, 
          edge_width = edge_width,
          edge_alpha = 1,
          edge_zorder = 1,
          scale = (scale[0] + 2, scale[1] + 2))

    if save_name is not None:
        plt.savefig(save_name, dpi = 300)
    else:
        plt.show()


def __scale(score, min_width, max_width, min_score = 0, max_score = 1):
    """
    Scales the score of the cause-effect relationship strength to a linewitdth

    Args:
        score (float): score to scale
        min_width (float): minimum linewidth
        max_width (float): maximum linewidth
        min_score (int, optional): minimum score range. Defaults to 0.
        max_score (int, optional): maximum score range. Defaults to 1.

    Returns:
        float: scaled score
    """
    return ((score-min_score)/(max_score-min_score))*(max_width-min_width)+min_width


def __nudge(pos, x_shift, y_shift):
    """
    Shift a position by x- and y-shift

    Args:
        pos (tuple(float,float)): x,y coords to shift
        x_shift (_type_): x-shift
        y_shift (_type_): y-shift

    Returns:
        tuple(float,float): shifted position
    """
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}


def __convert_to_string_graph(graph_bool):
    """Converts the 0,1-based graph returned by PCMCI to a string array
    with links '-->'.
    Parameters
    ----------
    graph_bool : array
        0,1-based graph array output by PCMCI.
    Returns
    -------
    graph : array
        graph as string array with links '-->'.
    """
    graph = np.zeros(graph_bool.shape, dtype='<U3')
    graph[:] = ""
    # Lagged links
    graph[:,:,1:][graph_bool[:,:,1:]==1] = "-->"
    # Unoriented contemporaneous links
    graph[:,:,0][np.logical_and(graph_bool[:,:,0]==1, 
                                graph_bool[:,:,0].T==1)] = "o-o"
    # Conflicting contemporaneous links
    graph[:,:,0][np.logical_and(graph_bool[:,:,0]==2, 
                                graph_bool[:,:,0].T==2)] = "x-x"
    # Directed contemporaneous links
    for (i,j) in zip(*np.where(
        np.logical_and(graph_bool[:,:,0]==1, graph_bool[:,:,0].T==0))):
        graph[i,j,0] = "-->"
        graph[j,i,0] = "<--"
    return graph


def __apply_alpha(result, alpha):
    """
    Applies alpha threshold to the pcmci result

    Args:
        result (dict): pcmci result
        alpha (float): significance level

    Returns:
        ndarray: graph filtered by alpha 
    """
    mask = np.ones(result['p_matrix'].shape, dtype='bool')
    # Set all p-values of absent links to 1.
    result['p_matrix'][mask==False] == 1.
    # Threshold p_matrix to get graph
    graph_bool = result['p_matrix'] <= alpha
    # Convert to string graph representation
    graph = __convert_to_string_graph(graph_bool)
    
    return graph


def __PCMCIres_converter(result):
    """
    Re-elaborates the PCMCI result in a new dictionary

    Args:
        result (dict): pcmci result

    Returns:
        dict: pcmci result re-elaborated
    """
    res_dict = {f:list() for f in result['var_names']}
    N, lags = result['graph'][0].shape
    for s in range(len(result['graph'])):
        for t in range(N):
            for lag in range(lags):
                if result['graph'][s][t,lag] == '-->':
                    res_dict[result['var_names'][t]].append({SOURCE : result['var_names'][s],
                                                             SCORE : result['val_matrix'][s][t,lag],
                                                             PVAL : result['p_matrix'][s][t,lag],
                                                             LAG : lag})
    return res_dict

