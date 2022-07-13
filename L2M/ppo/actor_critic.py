import torch
import torch.nn as nn
from torch.distributions import Categorical
from EEN.message_pass import sum_efeat


class ActorCritic(nn.Module):

    def __init__(self, actor_class, critic_class, input_dim, hidden_dim, output_dim, num_layers, device):
        super(ActorCritic, self).__init__()
        self.output_dim = output_dim
        self.actor_net = actor_class(input_dim, hidden_dim, output_dim, num_layers)
        self.critic_net = critic_class(input_dim, hidden_dim, 1, num_layers)
        self.device = device
        self.to(device)

    def get_masks_idxs_subg_h(self, ob, g):
        edge_mask = (ob.select(2, 0).long() == 0)
        flatten_edge_idxs = torch.nonzero(edge_mask.view(-1), as_tuple=False).squeeze(1)

        subg_mask = edge_mask.any(dim=1)
        flatten_subg_idxs = torch.nonzero(subg_mask, as_tuple=False).squeeze(1)

        subg_edge_mask = edge_mask.index_select(0, flatten_subg_idxs)
        flatten_subg_edge_idxs = torch.nonzero(subg_edge_mask.view(-1), as_tuple=False).squeeze(1)

        subg = g.edge_subgraph(subg_mask, relabel_nodes=False).to(self.device)
        h = self._build_h(ob).index_select(0, flatten_subg_idxs)

        return ((edge_mask, subg_mask, subg_edge_mask), (flatten_edge_idxs, flatten_subg_idxs, flatten_subg_edge_idxs), subg, h)

    def act(self, ob, g):
        num_edges, num_samples = ob.size(0), ob.size(1)
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        _, _, subg_edge_mask = masks
        flatten_edge_idxs, _, flatten_subg_edge_idxs = idxs

        logits = self.actor_net(h, subg, subg_edge_mask).view(-1, self.output_dim).index_select(0, flatten_subg_edge_idxs)

        m = Categorical(logits=logits)
        action = torch.zeros(num_edges * num_samples, dtype=torch.long, device=self.device)
        action[flatten_edge_idxs] = m.sample()
        action = action.view(-1, num_samples)
        return action

    def act_and_crit(self, ob, g):
        num_edges, num_samples = ob.size(0), ob.size(1)
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        _, _, subg_edge_mask = masks
        flatten_edge_idxs, _, flatten_subg_edge_idxs = idxs

        logits = self.actor_net(h, subg, subg_edge_mask).view(-1, self.output_dim).index_select(0, flatten_subg_edge_idxs)
        m = Categorical(logits=logits)

        action = torch.zeros(num_edges * num_samples, dtype=torch.long, device=self.device)
        action[flatten_edge_idxs] = m.sample()

        action_log_probs = torch.zeros(num_edges * num_samples, device=self.device).double()
        action_log_probs[flatten_edge_idxs] = m.log_prob(action.index_select(0, flatten_edge_idxs))
        action = action.view(-1, num_samples)
        action_log_probs = action_log_probs.view(-1, num_samples)

        edge_value_preds = torch.zeros(num_edges * num_samples, device=self.device).double()
        edge_value_preds[flatten_edge_idxs] = (self.critic_net(h, subg,
                                                               subg_edge_mask).view(-1).index_select(0, flatten_subg_edge_idxs))
        value_pred = sum_efeat(g.to(self.device), edge_value_preds.view(-1, num_samples)) / num_edges
        return action, action_log_probs, value_pred

    def evaluate_batch(self, ob, g, action):
        num_edges, num_samples = ob.size(0), ob.size(1)
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        edge_mask, _, subg_edge_mask = masks
        flatten_edge_idxs, _, flatten_subg_edge_idxs = idxs

        logits = self.actor_net(h, subg, subg_edge_mask).view(-1, self.output_dim).index_select(0, flatten_subg_edge_idxs)
        m = Categorical(logits=logits)

        action_log_probs = torch.zeros(num_edges * num_samples, device=self.device).double()
        action_log_probs[flatten_edge_idxs] = m.log_prob(action.reshape(-1).index_select(0, flatten_edge_idxs))
        action_log_probs = action_log_probs.view(-1, num_samples)

        edge_value_preds = torch.zeros(num_edges * num_samples, device=self.device).double()
        edge_value_preds[flatten_edge_idxs] = (self.critic_net(h, subg,
                                                               subg_edge_mask).view(-1).index_select(0, flatten_subg_edge_idxs))

        edge_entropies = -torch.sum(torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1)
        avg_entropy = edge_entropies.mean()

        value_preds = sum_efeat(g.to(self.device), edge_value_preds.view(-1, num_samples)) / num_edges
        return action_log_probs, avg_entropy, value_preds, edge_mask

    def _build_h(self, ob):
        ob_t = ob.select(2, 1).unsqueeze(2)
        ob_w = ob.select(2, 2).unsqueeze(2)
        return torch.cat([ob_t, ob_w], dim=2)
