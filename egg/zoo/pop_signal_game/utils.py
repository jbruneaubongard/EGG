import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime as dt
import json
import os 

def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {"acc": acc}


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}

def community_with_central_agent(nb_agents, weight_central_sender, weight_central_receiver, weight_noncentral, type_graph):
    """
    Returns the adjacency matrix of a community of nb_agents in which the central agent has index 0
    adjacency_matrix[i,j] is the weight of the interaction (sender=agent_i, receiver=agent_j)
    """
    if type_graph == 'neighbors':
        noncentral_agents = np.eye(nb_agents - 1, k=1, dtype=int) + np.eye(nb_agents - 1, k=-1, dtype=int)
        noncentral_agents[0,-1] = 1
        noncentral_agents[-1,0] = 1
        noncentral_agents *= weight_noncentral
    
    elif type_graph == 'fully_connected':
        noncentral_agents = np.ones((nb_agents-1, nb_agents-1)) - np.eye(nb_agents-1)
        noncentral_agents *= weight_noncentral
        
    first_row = np.array([[weight_central_sender for i in range(nb_agents-1)]], dtype=int)
    adjacency_matrix = np.concatenate((first_row,noncentral_agents), axis=0)

    first_column = np.array([[weight_central_receiver if i!=0 else 0 for i in range(nb_agents)]], dtype=int).T
    adjacency_matrix = np.concatenate((first_column,adjacency_matrix), axis=1)
    
    return adjacency_matrix

def two_communities(nb_agents, weight_central_sender, weight_central_receiver, weight_noncentral, bridge_agent):

    if not bridge_agent:
        assert (nb_agents % 2) == 0, "Odd number of agents in experiment 2, expected even"
        nb_agents_subcommunity = nb_agents//2
        
    else:
        assert (nb_agents % 2) == 1, "Even number of agents in experiment 3, expected odd"
        nb_agents_subcommunity = (nb_agents - 1)//2
    
    adjacency_matrix_subcommunity = community_with_central_agent(
        nb_agents_subcommunity, 
        weight_central_sender, 
        weight_central_receiver, 
        weight_noncentral,
        )

    if not bridge_agent:
        adjacency_matrix = np.block(
            [
            [adjacency_matrix_subcommunity, np.zeros_like(adjacency_matrix_subcommunity, dtype=int)],
            [np.zeros_like(adjacency_matrix_subcommunity, dtype=int), adjacency_matrix_subcommunity],
            ], 
        dtype=int,
        )
    
    else:
        adjacency_matrix = np.block(
            [
            [adjacency_matrix_subcommunity, np.zeros((nb_agents_subcommunity, nb_agents_subcommunity + 1), dtype=int)],
            [np.zeros((1,nb_agents_subcommunity), dtype=int), np.zeros((1,nb_agents_subcommunity + 1), dtype=int)],
            [np.zeros((nb_agents_subcommunity, nb_agents_subcommunity + 1), dtype=int), adjacency_matrix_subcommunity],
            ], 
        dtype=int,
        )

    return adjacency_matrix

def get_folder_saved_data(opts):
    """
    Returns the path to the folder where the results will be stored, characterized by the date and (sub)type of experiment
    """
    if opts.save_data:
        date_exp = dt.datetime.today().strftime("%Y-%m-%d_%Hh%M")
        if opts.type_exp=='training':
            folder = f"{opts.path_save_exp_data}/{date_exp}_{opts.type_exp}_nb-agents={opts.nb_agents}"
        else:
            folder = f"{opts.path_save_exp_data}/{date_exp}_{opts.type_exp}_{opts.subtype_exp}_nb-agents={opts.nb_agents}"
        return folder
    else:
        return ""

def save_agents(list_agents, folder_saved_data):
    for i in range(len(list_agents)):
        torch.save(list_agents[i], f"{folder_saved_data}/agent_{i}.pt")
    return None

def get_new_communities_from_saved_agents(opts, folder_saved_data):
    """
    From a folder where agents were created during training, creates a list of new communities, the central agent
    coming each time from a different training community.
    Outputs a dictionary stating the origin of each agent.
    Saves agents' data of the new community (before exp)
    """
    lists_of_agents = []
    dict_origin_of_agents = {str(i) : [] for i in range(opts.nb_agents)}

    for idx_com_central_agent in range(opts.nb_agents):
        # First agent (index 0 of the new list) is the central agent of the new community
        agents = [torch.load(f"{opts.path_agents_data}/community={idx_com_central_agent}/agent_{idx_com_central_agent}.pt")]
        dict_origin_of_agents[str(idx_com_central_agent)].append(idx_com_central_agent)
        
        # Add other agents, each one from the remaining communities
        for i in range(opts.nb_agents):
            if i!= idx_com_central_agent:
                agents.append(torch.load(f"{opts.path_agents_data}/community={i}/agent_{idx_com_central_agent}.pt"))
                dict_origin_of_agents[str(idx_com_central_agent)].append(i)
        
        # Save agents' weights in new folder
        folder_com_data = folder_saved_data + '/new_community=' + str(idx_com_central_agent) + '/initial'
        if not os.path.exists(folder_com_data):
            os.makedirs(folder_com_data)
        
        save_agents(agents, folder_com_data)

        #
        lists_of_agents.append(nn.ModuleList(agents))

    return lists_of_agents, dict_origin_of_agents

def merge_communities_from_saved_agents(opts, folder_saved_data):
    """
    From a folder where agents were created during training, creates a list of new graphs, 
    merging each time two different communities.
    Outputs a dictionary stating the origin of agents in the new graphs.
    Saves agents' data of the new community (before exp)
    """
    lists_of_agents = []
    dict_origin_of_agents = {str(i) : [] for i in range(opts.nb_agents*(opts.nb_agents-1)//2)}

    idx_graph = 0

    for idx_first_com in range(opts.nb_agents):
        for idx_second_com in range(idx_first_com + 1, opts.nb_agents):

            # Merge two communities 

            first_com = [torch.load(f"{opts.path_agents_data}/community={idx_first_com}/agent_{i}.pt") for i in range(opts.nb_agents)]
            second_com = [torch.load(f"{opts.path_agents_data}/community={idx_second_com}/agent_{i}.pt") for i in range(opts.nb_agents)]

            agents = first_com + second_com

            dict_origin_of_agents[str(idx_graph)] += [idx_first_com, idx_second_com]

            # Save agents' weights in new folder
            folder_com_data = folder_saved_data + '/graph=' + str(idx_graph) + '/initial'
            if not os.path.exists(folder_com_data):
                os.makedirs(folder_com_data)
            
            save_agents(agents, folder_com_data)

            #
            lists_of_agents.append(nn.ModuleList(agents))

            idx_graph += 1

    return lists_of_agents, dict_origin_of_agents

def merge_communities_with_bridge_agent(opts, folder_saved_data):
    """
    From a folder where agents were created during training, creates a list of new graphs, 
    merging each time two different communities and introducing a bridge agent from another 
    community (sampled randomly from this other community).
    Outputs a dictionary stating the origin of agents in the new graphs.
    Saves agents' data of the new community (before exp)
    """
    lists_of_agents = []
    nb_new_graphs = (opts.nb_agents*(opts.nb_agents-1)//2)*(opts.nb_agents-2)
    dict_origin_of_agents = {str(i) : [] for i in range(nb_new_graphs)}

    idx_graph = 0

    for idx_first_com in range(opts.nb_agents):
        for idx_second_com in range(idx_first_com + 1, opts.nb_agents):
            for idx_bridge_com in range(opts.nb_agents):
                if idx_bridge_com != idx_first_com and idx_bridge_com != idx_second_com:

                    # Merge two communities and add bridge agent

                    first_com = [torch.load(f"{opts.path_agents_data}/community={idx_first_com}/agent_{i}.pt") for i in range(opts.nb_agents)]
                    idx_bridge_agent = np.random.randint(opts.nb_agents)
                    bridge_agent = [torch.load(f"{opts.path_agents_data}/community={idx_first_com}/agent_{idx_bridge_agent}.pt")]
                    second_com = [torch.load(f"{opts.path_agents_data}/community={idx_second_com}/agent_{i}.pt") for i in range(opts.nb_agents)]

                    agents = first_com + bridge_agent + second_com

                    dict_origin_of_agents[str(idx_graph)] += [idx_first_com, (idx_bridge_com, idx_bridge_agent), idx_second_com]

                    # Save agents' weights in new folder
                    folder_com_data = folder_saved_data + '/graph=' + str(idx_graph) + '/initial'
                    if not os.path.exists(folder_com_data):
                        os.makedirs(folder_com_data)
                    
                    save_agents(agents, folder_com_data)

                    #
                    lists_of_agents.append(nn.ModuleList(agents))

                    idx_graph += 1

    return lists_of_agents, dict_origin_of_agents


class NumpyEncoder(json.JSONEncoder):
    """ 
    Special json encoder for numpy types 
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)