import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.zoo.pop_signal_game.utils import loss, loss_nll, community_with_central_agent, two_communities
from egg.zoo.pop_signal_game.reinforce_wrappers import SymbolPopGameReinforce
from egg.zoo.pop_signal_game.gs_wrappers import SymbolPopGameGS

def get_game(senders, receivers, adjacency_matrix, opt):
    idx_list = list(iter(AgentIndexIterator(adjacency_matrix)))
        
    if opt.mode == "rf":
        game = SymbolPopGameReinforce(
            senders,
            receivers,
            idx_list,
            loss,
            sender_entropy_coeff=0.01,
            receiver_entropy_coeff=0.01,
        )

    elif opt.mode == "gs":
        game = SymbolPopGameGS(
            senders,
            receivers,
            idx_list,
            opt.gs_tau,
            loss_nll,
        )

    else:
        raise RuntimeError(f"Unknown training mode: {opt.mode}")

    return game

    
class Graph:
    def __init__(
        self,
        opts,
        ):

        assert opts.type_exp in ['training', 'exp_1', 'exp_2', 'exp_3'], "The experiment type is not valid"
        self.type_experiment = opts.type_exp
        

        if self.type_experiment == 'training':
            """
            Array of shape (nb_agents, nb_agents) filled with ones, except from the diagonal (filled with zeros). 
            Corresponds to a fully connected graph
            """
            if opts.nb_agents==1:
                self.adjacency_matrix = np.array([[1]])
            else:
                self.adjacency_matrix = np.ones((opts.nb_agents, opts.nb_agents), dtype=int) - np.eye(opts.nb_agents, dtype=int)


        else:
            self.weight_central_sender = opts.w_central_sender
            self.weight_central_receiver = opts.w_central_receiver
            self.weight_noncentral = opts.w_noncentral
        

        if self.type_experiment == 'exp_1':
            assert opts.subtype_exp in ['neighbors', 'fully_connected'], "The experiment subtype is not valid"
            self.subtype_experiment = opts.subtype_exp
            
            self.adjacency_matrix = community_with_central_agent(
                opts.nb_agents, 
                self.weight_central_sender, 
                self.weight_central_receiver, 
                self.weight_noncentral,
                self.subtype_experiment,
                )
            
        if self.type_experiment == 'exp_2':

            if opts.bridge_w == 0:
                print("WARNING: bridge weight is equal to zero")
                further = input("Continue? (y/n): ")
                if further == 'n':
                    sys.exit("bridge weight was equal to zero")

            self.bridge_weight = opts.bridge_w
            assert opts.subtype_exp in ['central-central', 'central-noncentral', 'noncentral-noncentral'], "The experiment subtype is not valid"
            self.subtype_experiment = opts.subtype_exp

            adjacency_matrix = two_communities(
                opts.nb_agents, 
                self.weight_central_sender, 
                self.weight_central_receiver, 
                self.weight_noncentral, 
                bridge_agent=False,
                )

            if self.subtype_experiment == 'central-central':
                adjacency_matrix[0, opts.nb_agents//2] = self.bridge_weight
                adjacency_matrix[opts.nb_agents//2, 0] = self.bridge_weight

            if self.subtype_experiment == 'central-noncentral':
                adjacency_matrix[0, opts.nb_agents//2 + 1] = self.bridge_weight
                adjacency_matrix[opts.nb_agents//2 + 1, 0] = self.bridge_weight
            
            if self.subtype_experiment == 'noncentral-noncentral':
                adjacency_matrix[1, opts.nb_agents//2] = self.bridge_weight
                adjacency_matrix[opts.nb_agents//2, 1] = self.bridge_weight

            self.adjacency_matrix = adjacency_matrix
        

        if self.type_experiment == 'exp_3':

            if opts.w_bridge_sender == 0 or opts.w_bridge_receiver == 0:
                print("WARNING: bridge sender or receiver weight is equal to zero")
                further = input("Continue? (y/n): ")
                if further == 'n':
                    sys.exit("bridge sender or receiver weight was equal to zero")

            assert opts.subtype_exp in ['central-central', 'central-noncentral', 'noncentral-noncentral'], "The experiment subtype is not valid"
            self.subtype_experiment = opts.subtype_exp

            adjacency_matrix = two_communities(
                opts.nb_agents, 
                self.weight_central_sender, 
                self.weight_central_receiver, 
                self.weight_noncentral, 
                bridge_agent=True,
                )
            
            if self.subtype_experiment == 'central-central':
                adjacency_matrix[opts.nb_agents//2, 0] = opts.w_bridge_sender
                adjacency_matrix[opts.nb_agents//2, opts.nb_agents//2 + 1] = opts.w_bridge_sender
                
                adjacency_matrix[0, opts.nb_agents//2] = opts.w_bridge_receiver
                adjacency_matrix[opts.nb_agents//2 + 1, opts.nb_agents//2] = opts.w_bridge_receiver

            if self.subtype_experiment == 'central-noncentral':
                adjacency_matrix[opts.nb_agents//2, 0] = opts.w_bridge_sender
                adjacency_matrix[opts.nb_agents//2, opts.nb_agents//2 + 2] = opts.w_bridge_sender
                
                adjacency_matrix[0, opts.nb_agents//2] = opts.w_bridge_receiver
                adjacency_matrix[opts.nb_agents//2 + 2, opts.nb_agents//2] = opts.w_bridge_receiver
            
            if self.subtype_experiment == 'noncentral-noncentral':
                adjacency_matrix[opts.nb_agents//2, 1] = opts.w_bridge_sender
                adjacency_matrix[opts.nb_agents//2, opts.nb_agents//2 + 2] = opts.w_bridge_sender
                
                adjacency_matrix[1, opts.nb_agents//2] = opts.w_bridge_receiver
                adjacency_matrix[opts.nb_agents//2 + 2, opts.nb_agents//2] = opts.w_bridge_receiver

            self.adjacency_matrix = adjacency_matrix



class AgentIndexIterator:

    """Creates an iterator over the indices of sender/receiver pairs according to the adjacency matrix of the graph provided. 
    For instance:
    np.array([[0, 1, 2],
              [0, 0, 3],
              [2, 0 , 1]])
    --> iterator over [(0,1), (0,2), (0,2), (1,2), (1,2), (1,2), (2,0), (2,0), (2,2)]
    Weights in the adjacency matrix must be integers
    """

    def __init__(
        self,
        adjacency_matrix,
        ):
        self.adjacency_matrix = adjacency_matrix
        self.nonzero_weights = np.nonzero(self.adjacency_matrix)
        self.current_idx = 0
        self.count = 0

    def __iter__(self):
        if not np.any(self.adjacency_matrix):
            raise StopIteration
        else:
            self.a = (self.nonzero_weights[0][self.current_idx], self.nonzero_weights[1][self.current_idx])
            self.count = int(self.adjacency_matrix[self.a[0], self.a[1]])
            
            assert self.count >=0, "Negative weight"

            self.count -= 1

            return self

    def __next__(self):
        x = self.a
        assert self.count >=0, "Negative weight"

        if self.count > 0: #count one sender-receiver interaction
            self.count -=1

        elif self.count == 0: #change the sender-receiver pair
            self.current_idx += 1

            if self.current_idx == len(self.nonzero_weights[0]):
                pass

            elif self.current_idx > len(self.nonzero_weights[0]):
                raise StopIteration

            else:
                self.a = (self.nonzero_weights[0][self.current_idx], self.nonzero_weights[1][self.current_idx])
                self.count = self.adjacency_matrix[self.a[0], self.a[1]]
                self.count -=1
                
        return x


class Agent(nn.Module):
    """ Each agent has a sender and receiver module. """ 
    def __init__(
        self,
        opt,
        feat_size=4096
    ):
        super(Agent, self).__init__()
        self.lin1 = nn.Linear(feat_size, opt.embedding_size, bias=False)
        self.sender = Sender(
            opt.game_size, 
            opt.embedding_size, 
            opt.hidden_size, 
            self.lin1, 
            opt.vocab_size, 
            opt.tau_s,
            )
        self.receiver = Receiver(
            opt.game_size, 
            opt.embedding_size, 
            opt.vocab_size, 
            opt.mode == 'rf',
            self.lin1,
            )


class Sender(nn.Module):
    def __init__(
        self,
        game_size,
        embedding_size,
        hidden_size,
        lin1,
        vocab_size,
        temperature,
    ):
        super(Sender, self).__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temperature = temperature

        self.lin1 = lin1
        self.conv2 = nn.Conv2d(
            1,
            hidden_size,
            kernel_size=(game_size, 1),
            stride=(game_size, 1),
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False
        )
        self.lin4 = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x, _aux_input=None):
        emb = self.return_embeddings(x)

        # in: h of size (batch_size, 1, game_size, embedding_size)
        # out: h of size (batch_size, hidden_size, 1, embedding_size)
        h = self.conv2(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.conv3(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        # h of size (batch_size, embedding_size)
        h = self.lin4(h)
        h = h.mul(1.0 / self.temperature)
        # h of size (batch_size, vocab_size)
        logits = F.log_softmax(h, dim=1)
        
        return logits

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x 1 x embedding_size
            embs.append(h_i)
        # concatenate the embeddings
        h = torch.cat(embs, dim=2)

        return h


class Receiver(nn.Module):
    def __init__(
            self,
            game_size,
            embedding_size,
            vocab_size,
            reinforce,
            lin1
    ):
        super(Receiver, self).__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size

        self.lin1 = lin1
        if reinforce:
            self.lin2 = nn.Embedding(vocab_size, embedding_size)
        else:
            self.lin2 = nn.Linear(vocab_size, embedding_size, bias=False)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h