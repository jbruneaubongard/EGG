import argparse
import os
import json

import torch
import torch.nn as nn

import egg.core as core
from egg.zoo.pop_signal_game.features import ImageNetFeat, ImagenetLoader
from egg.zoo.pop_signal_game.archs import Graph, Agent, get_game
from egg.zoo.pop_signal_game.utils import save_agents, get_folder_saved_data
from egg.zoo.pop_signal_game.trainers import Trainer

def parse_arguments():
    parser = argparse.ArgumentParser()
    #change default when publishing
    parser.add_argument(
        "--root", 
        default="/homedtcl/jbruneaubongard/EGG/data/signaling_game_data/", 
        help="data root folder",
        )
    parser.add_argument(
        "--nb_agents", 
        type=int, 
        default=1,
        help="Number of agents in the training population",
        )
    parser.add_argument(
        "--type_exp", 
        default='training', 
        help="Type of experiment: training, exp_1, exp_2, exp_3",
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

    # Trains N=nb_agents communities of N fully connected agents each
    for idx_community in range(opts.nb_agents):

        # Generates random agents (initial agents to be trained)
        agents = nn.ModuleList([Agent(opts) for i in range(opts.nb_agents)])
        
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
        
        features_to_save = ['root', 'nb_agents', 'type_exp', 'path_save_exp_data', 
                            'mode', 'tau_s', 'game_size', 'same', 
                            'embedding_size', 'hidden_size', 
                            'n_epochs', 'batches_per_epoch', 'gs_tau']
        
        folder_com_data = folder_saved_data + '/community=' + str(idx_community)
        if not os.path.exists(folder_com_data):
            os.makedirs(folder_com_data)
            
        # Save hyperparameters
        with open(folder_com_data + '/args.txt', 'w') as f:
            dict_opts = {k: v for (k,v) in vars(opts).items() if k in features_to_save}
            json.dump(dict_opts, f, indent=2)

        trainer.train(n_epochs=opts.n_epochs, path=folder_com_data)
        
        # Save agents' weights
        if opts.save_data:
            save_agents(agents, folder_com_data)
    
    core.close()

