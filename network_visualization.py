from graphviz import Digraph

def plot_network(genotypes, file_name = None,save_path = None, format = 'jpg'):
    
    cell_num = genotypes['cell_num']
    
    reduction_list = [cell_num//3, 2*cell_num//3]
    
    computing_node_num = genotypes['computing_node_num']
    primitives = genotypes['primitives_finally']
    indicators = genotypes['indicators_finally']
    
    g = Digraph(format = format, edge_attr = dict(fontsize='20', weight='2'),
                node_attr = dict(style = 'filled', align = 'center', fontsize = '20', penwidth = '3'), engine = 'dot')
    
    line = 7
    c_node = '0.60'
    o_node = '0.80'
    out_node = '2.0'
    fontsize = '30'
    c_color = 'lightblue'
    o_color = 'lemonchiffon'
    weight = '1.0'
    
    g.body.extend(['rankdir=LR'])
    
    g.node(name = 'stem_0', label = '', fillcolor = 'lightgreen', height = c_node, width = o_node, style = 'filled', shape = 'circle')
    g.node(name = 'stem_1', label = '', fillcolor = 'lightgreen', height = c_node, width = o_node, style = 'filled', shape = 'circle')         

    g.node(name = 'Input', fillcolor = 'lightgreen', height = out_node, width = out_node,
           fontname = 'Microsoft Yahei', fontsize = fontsize, style = 'filled', shape='circle')        
    g.edge('Input', 'stem_0', penwidth = str(line), weight = weight, arrowsize = '0.6')
    g.edge('Input', 'stem_1', penwidth = str(line), weight = weight, arrowsize = '0.6')

    def decide_wheather_to_plot(cell_order, total_cell_num = 14):
        plot = False
        if cell_order < total_cell_num - 1:
            for computing_node_order in range(computing_node_num):
                next_node_name = 'cell_{},node_{}'.format(cell_order + 1, computing_node_order + 2)
                next_node_indicators = indicators[next_node_name]
                if 0 in next_node_indicators:
                    plot = True
        if cell_order < total_cell_num - 2:
            for computing_node_order in range(computing_node_num):
                next_node_name = 'cell_{},node_{}'.format(cell_order + 2, computing_node_order + 2)
                next_node_indicators = indicators[next_node_name]
                if 1 in next_node_indicators:
                    plot = True
        if cell_order == total_cell_num - 1:
            plot = True
        return plot
            
            
            
    for cell_order in range(cell_num):
        
        plot = decide_wheather_to_plot(cell_order, total_cell_num = cell_num)
        if plot:
            if cell_order in reduction_list:
                c_color = 'red'
            else:
                c_color = 'lightblue'
            
            output_name = 'cell_{},node_6'.format(cell_order)
            
            g.node(name = output_name, label = '', fillcolor = o_color, height = o_node, width = o_node, shape = 'circle')
            
            for computing_node_order in range(computing_node_num):
                node_name = 'cell_{},node_{}'.format(cell_order, computing_node_order + 2)
                 
                g.node(name = node_name, label = '', fillcolor = c_color, height = c_node, width = c_node, shape = 'circle')
                           
                g.edge(node_name, output_name, color = 'black', style = "dotted", penwidth = str(int(1.3*line)), weight = weight, arrowsize = '0.5')
                
                node_indicators = indicators[node_name]
                node_primitives = primitives[node_name]
    
                for order in range(len(node_indicators)):
                    
                    input_id = node_indicators[order]
                    input_primitive = node_primitives[order]
                    
                    if input_primitive == 'none':
                        pass
                    else:                
                        input_color = get_color(input_primitive)
                        line_width = get_line_width(input_primitive, line)
                        
                        if cell_order == 0:
                            if input_id in [0, 1]:
                                input_name = 'stem_{}'.format(input_id)
                            else:
                                input_name = 'cell_0,node_{}'.format(input_id)
                        elif cell_order == 1:
                            if input_id == 0:
                                input_name = 'cell_0,node_6'
                            elif input_id == 1 :
                                input_name = 'stem_0'
                            else:
                                input_name = 'cell_1,node_{}'.format(input_id)
                        else:
                            if input_id in [0, 1]:
                                input_name = 'cell_{},node_6'.format(cell_order - input_id - 1)
                            else:
                                input_name = 'cell_{},node_{}'.format(cell_order , input_id)
                        g.edge(input_name, node_name, color = input_color, penwidth = line_width, weight = weight, arrowsize = '0.6')

    g.node(name = 'Output', label = 'Output', fillcolor = 'lightgreen', height = out_node, width = out_node, fontname = 'Microsoft Yahei',
               style = 'filled', fontsize = fontsize, shape = 'circle')
    g.edge('cell_{},node_6'.format(cell_num - 1), 'Output', penwidth = str(line), weight = weight, color = 'black', style = "dotted", arrowsize = '0.6')
        
    if save_path is None:
        g.view()
    else:
        g.render(file_name, save_path)
        del g
    

def get_color(primitive):
    if primitive == 'sep_conv_3':
        color = 'blue'
    elif primitive == 'sep_conv_5':
        color = 'darkblue'
    elif primitive == 'dil_conv_3':
        color = 'olive'
    elif primitive == 'dil_conv_5':
        color = 'darkgreen'
    elif primitive == 'max_pool_3':
        color = 'peru'
    elif primitive == 'avg_pool_3':
        color = 'orange'
    elif primitive == 'skip_connect':
        color = 'red'
    elif primitive == 'none':
        color = 'slategrey'
    return color        

def get_line_width(primitive, standard):
    if primitive == 'sep_conv_3':
        width = int(1.0*standard)
    elif primitive == 'sep_conv_5':
        width = int(1.3*standard)
    elif primitive == 'dil_conv_3':
        width = standard
    elif primitive == 'dil_conv_5':
        width = int(1.3*standard)
    elif primitive == 'max_pool_3':
        width = int(0.9*standard)
    elif primitive == 'avg_pool_3':
        width = int(0.9*standard)
    elif primitive == 'skip_connect':
        width = int(0.8*standard)
    elif primitive == 'none':
        width = int(0.8*standard)
    return str(width)

