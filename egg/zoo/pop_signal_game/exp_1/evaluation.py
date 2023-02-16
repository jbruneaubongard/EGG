import argparse
import os

import torch
import pandas as pd

import egg.core as core
from egg.zoo.signal_game.features import ImageNetFeat, ImagenetLoader
from egg.zoo.pop_signal_game.utils import loss, loss_nll

def parse_arguments():
    parser = argparse.ArgumentParser()
    #change default when publishing
    parser.add_argument(
        "--root", 
        default="/homedtcl/jbruneaubongard/EGG/data/signaling_game_data/", 
        help="data root folder",
        )
    parser.add_argument(
        "--path_agents_data",
        default="",
        help="Path to the directory where agents' networks are saved from training.",
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

def get_agents(path):
    """
    Takes an input the folder where data from exp are stored
    Outputs a list of agents, where agents[com][nb][0] is the initial agent nb of community com
                                    agents[com][nb][1] is the final agent nb of community com
    """
    nb_com = len([name for name in os.listdir(path)])
    nb_agents = len([name for name in os.listdir(f'{path}/new_community=0/initial')])

    agents = [[[
    torch.load(f'{path}/new_community={idx_com}/initial/agent_{idx_agent}.pt'), 
    torch.load(f'{path}/new_community={idx_com}/final/agent_{idx_agent}.pt')
    ] for idx_agent in range(nb_agents)
    ] for idx_com in range(nb_com)
    ]

    return agents

def get_game(sender, receiver, opt):
    if opts.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=0.01,
            receiver_entropy_coeff=0.01,
        )
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game

def get_success_rate(sender, receiver, test_loader, opt):
    
    game = get_game(sender, receiver, opt)
    optimizer = core.build_optimizer(game.parameters())
    callback = None
    if opts.mode == "gs":
        callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []

    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=None,
        validation_data=None,
        callbacks=callbacks,
    )
    mean_loss, full_interaction = trainer.eval(data=test_loader)
    return mean_loss, full_interaction.aux['acc'].mean().item()

def cross_time_evaluation(agents, test_loader, opt, df):
    """
    Returns a DataFrame filled with the average success rates / losses between the same initial and final agents
    """
    for idx_com in range(len(agents)):
        for idx_agent in range(len(agents[idx_com])):
            # The initial agent is the sender
            mean_loss, mean_acc = get_success_rate(agents[idx_com][idx_agent][0].sender, agents[idx_com][idx_agent][1].receiver, test_loader, opt)
            df.loc[len(df.index)] = [idx_com, idx_agent, 'initial', idx_agent, 'final', mean_loss, mean_acc] 

            # The initial agent is the receiver
            mean_loss, mean_acc = get_success_rate(agents[idx_com][idx_agent][1].sender, agents[idx_com][idx_agent][0].receiver, test_loader, opt)
            df.loc[len(df.index)] = [idx_com, idx_agent, 'final', idx_agent, 'initial', mean_loss, mean_acc] 

    return df

def same_time_evaluation(agents, test_loader, opt, df):
    """
    Returns a DataFrame filled with the average success rates / losses between the central agent and each noncentral agent in the same state (initial/final)
    """
    for idx_com in range(len(agents)):
        for idx_agent in range(len(agents[idx_com])):
            # Looking at initial agents

            ## The central agent is the sender
            mean_loss, mean_acc = get_success_rate(agents[idx_com][0][0].sender, agents[idx_com][idx_agent][0].receiver, test_loader, opt)
            df.loc[len(df.index)] = [idx_com, 0, 'initial', idx_agent, 'initial', mean_loss, mean_acc]

            ## The central agent is the receiver
            mean_loss, mean_acc = get_success_rate(agents[idx_com][idx_agent][0].sender, agents[idx_com][0][0].receiver, test_loader, opt)
            df.loc[len(df.index)] = [idx_com, idx_agent, 'initial', 0, 'initial', mean_loss, mean_acc]

            # Looking at final agents

            ## The central agent is the sender
            mean_loss, mean_acc = get_success_rate(agents[idx_com][0][1].sender, agents[idx_com][idx_agent][1].receiver, test_loader, opt)
            df.loc[len(df.index)] = [idx_com, 0, 'final', idx_agent, 'final', mean_loss, mean_acc]

            ## The central agent is the receiver
            mean_loss, mean_acc = get_success_rate(agents[idx_com][idx_agent][1].sender, agents[idx_com][0][1].receiver, test_loader, opt)
            df.loc[len(df.index)] = [idx_com, idx_agent, 'final', 0, 'final', mean_loss, mean_acc]

    return df

if __name__ == "__main__":
    opts = parse_arguments()

    data_folder = os.path.join(opts.root, "test/")
    dataset = ImageNetFeat(root=data_folder)

    test_loader = ImagenetLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        opt=opts,
        batches_per_epoch=opts.batches_per_epoch,
        seed=None,
    )
    d = {
        'com_id': [],
        'sender_id': [],
        'sender_state': [],
        'receiver_id': [],
        'receiver_state': [],
        'mean_loss': [],
        'mean_accuracy': []
    }
    df = pd.DataFrame(d)

    agents = get_agents(opts.path_agents_data)

    df = cross_time_evaluation(agents, test_loader, opts, df)
    df = same_time_evaluation(agents, test_loader, opts, df)
    
    df.to_csv(f'{opts.path_agents_data}/evaluation.csv', index=False)
    
    core.close()