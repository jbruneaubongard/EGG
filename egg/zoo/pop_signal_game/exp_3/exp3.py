import argparse
import os
import json

import torch.nn as nn

import egg.core as core
from egg.zoo.pop_signal_game.features import ImageNetFeat, ImagenetLoader
from egg.zoo.pop_signal_game.archs import Graph, get_game
from egg.zoo.pop_signal_game.utils import save_agents, get_folder_saved_data, merge_communities_with_bridge_agent, NumpyEncoder
from egg.zoo.pop_signal_game.trainers import Trainer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", 
        default="", 
        help="data root folder",
        )
    parser.add_argument(
        "--nb_agents", 
        type=int, 
        default=1,
        help="Number of agents in the training population\
        For exp2 and exp3, number of agents in each training community",
        )
    parser.add_argument(
        "--type_exp", 
        default='exp_3', 
        help="Type of experiment: training, exp_1, exp_2, exp_3",
        )
    parser.add_argument(
        "--subtype_exp", 
        default='', 
        help="Subtype of experiment. \
        If exp_1: neighbors or fully_connected. \
        If exp_2 or exp_3: central-central, central-noncentral or noncentral-noncentral",
        )
    parser.add_argument(
        "--w_central_sender", 
        type=int, 
        default=0,
        help="Sender weight of the central agent",
        )
    parser.add_argument(
        "--w_central_receiver", 
        type=int, 
        default=0,
        help="Receiver weight of the central agent",
        )
    parser.add_argument(
        "--w_noncentral", 
        type=int, 
        default=0,
        help="Weight of interaction between two noncentral nodes",
        )
    parser.add_argument(
        "--w_bridge_sender", 
        type=int, 
        default=0,
        help="Sender weight of the bridge agent (exp_3)",
        )
    parser.add_argument(
        "--w_bridge_receiver", 
        type=int, 
        default=0,
        help="Receiver weight of the bridge agent (exp_3)",
        )
    parser.add_argument(
        "--save_data",
        type=bool,
        default=False,
        help="Whether to save exp data or not",
        )
    parser.add_argument(
        "--path_save_exp_data",
        default="/results",
        help="Path to the directory where agents' networks will be saved at the end of the process",
        )
    parser.add_argument(
        "--path_agents_data",
        default="",
        help="Path to the directory where agents' networks are saved from training. \
            They will be used to create the agents list.",
        )
    parser.add_argument(
        "--tau_s", 
        type=float, 
        default=10.0, 
        help="Sender Gibbs temperature",
        )
    parser.add_argument(
        "--game_size", 
        type=int, 
        default=2, 
        help="Number of images seen by an agent",
        )
    parser.add_argument(
        "--same", 
        type=int, 
        default=0, 
        help="Use same concepts",
        )
    parser.add_argument(
        "--embedding_size", 
        type=int, 
        default=50, 
        help="embedding size",
        )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=20,
        help="hidden size (number of filters informed sender)",
        )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="Batches in a single training/validation epoch",
        )
    parser.add_argument(
        "--inf_rec", 
        type=int, 
        default=0, 
        help="Use informed receiver",
        )
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.",
        )
    parser.add_argument(
        "--gs_tau", 
        type=float, 
        default=1.0, 
        help="GS temperature",
        )


    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt

if __name__ == "__main__":

    opts = parse_arguments()

    folder_saved_data = get_folder_saved_data(opts) # returns "" if not opts.save_data
    
    lists_of_agents, dict_origin_of_agents = merge_communities_with_bridge_agent(opts, folder_saved_data)

    nb_graphs = (opts.nb_agents*(opts.nb_agents-1)//2)*(opts.nb_agents-2)

    for idx_graph in range(nb_graphs):

        agents = lists_of_agents[idx_graph]
    
        senders = nn.ModuleList([agent.sender for agent in agents])
        receivers = nn.ModuleList([agent.receiver for agent in agents])

        # interaction graph
        graph = Graph(opts)

        data_folder = os.path.join(opts.root, "train/")
        dataset = ImageNetFeat(root=data_folder)

        train_loader = ImagenetLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            opt=opts,
            batches_per_epoch=opts.batches_per_epoch,
            seed=None,
        )
        
        validation_loader = ImagenetLoader(
            dataset,
            opt=opts,
            batch_size=opts.batch_size,
            batches_per_epoch=opts.batches_per_epoch,
            seed=7,
        )

        game = get_game(senders, receivers, graph.adjacency_matrix, opts)

        optimizer = core.build_optimizer(game.parameters())
        callback = None
        if opts.mode == "gs":
            callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1) for sender in senders]
        else:
            callbacks = []

        callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))

        trainer = Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            opt=opts,
            validation_data=validation_loader,
            callbacks=callbacks,
        )
        
        features_to_save = ['root', 'nb_agents', 'type_exp', 'subtype_exp', 
                            'w_central_sender', 'w_central_receiver', 'w_noncentral', 
                            'w_bridge_sender', 'w_bridge_receiver',
                            'path_save_exp_data', 'path_agents_data', 
                            'mode', 'tau_s', 'game_size', 'same', 
                            'embedding_size', 'hidden_size', 'n_epochs', 'batches_per_epoch']

        folder_com_data = folder_saved_data + '/graph=' + str(idx_graph) 
        
        # Save hyperparameters
        with open(folder_com_data + '/args.txt', 'w') as f:
            dict_opts = {k: v for (k,v) in vars(opts).items() if k in features_to_save}
            dict_opts['origin_agents'] = dict_origin_of_agents[str(idx_graph)]
            dict_opts['adjacency_matrix'] = graph.adjacency_matrix
            json.dump(dict_opts, f, cls=NumpyEncoder, indent=2)

        trainer.train(n_epochs=opts.n_epochs, path=folder_com_data)
        
        # Save agents' weights
        if not os.path.exists(folder_com_data + '/final'):
            os.makedirs(folder_com_data + '/final')

        if opts.save_data:
            save_agents(agents, folder_com_data + '/final')
    
    core.close()

