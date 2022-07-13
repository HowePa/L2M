import torch
from EEN.message_pass import sum_efeat, update_efeat


class MaximumMatchingEnv(object):
    def __init__(self, max_epi_t, device):
        self.max_epi_t = max_epi_t
        self.device = device

    def step(self, action):
        reward, sol, done = self._take_action(action)

        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        return ob, reward, done, info

    def _take_action(self, action):
        self.t += 1
        self.x += action
        # exclude near
        x1 = (self.x == 1)
        x1_deg = update_efeat(self.g, x1.double())
        undetermined = (self.x == 0)
        excluded = undetermined & (x1_deg > 0)
        self.x[excluded] = -1
        # rollback
        clashed = x1 & (x1_deg > 0)
        self.x[clashed] = 0
        # timeout
        undetermined = (self.x == 0)
        timeout = (self.t == self.max_epi_t)
        self.x[undetermined & timeout] = -1

        done = self._check_done()
        # get reward
        x1 = (self.x == 1).float()
        h = self.w * x1
        
        next_sol = sum_efeat(self.g, h)
        reward = (next_sol - self.sol)
        reward /= self.edge_norm

        return reward, next_sol, done

    def _check_done(self):
        undecided = (self.x == 0).double()
        num_undecided = sum_efeat(self.g, undecided)
        done = (num_undecided == 0)

        return done

    def _build_ob(self):
        ob_x = self.x.unsqueeze(2).double()
        ob_t = self.t.unsqueeze(2).double() / self.max_epi_t
        ob_w = self.w.unsqueeze(2).double()
        ob = torch.cat([ob_x, ob_t, ob_w], dim=2)
        return ob

    def register(self, g, num_samples=1):
        self.g = g.to(self.device)
        self.num_samples = num_samples
        self.num_edges = self.g.number_of_edges()
        self.edge_norm = self.g.batch_num_edges().double().view(-1, 1)

        self.x = torch.zeros(self.num_edges, num_samples, dtype=torch.long, device=self.device)
        self.t = torch.zeros(self.num_edges, num_samples, dtype=torch.long, device=self.device)
        self.w = torch.cat([self.g.edata["weight"]] * num_samples, dim=-1).to(self.device)
        self.step_num = self.max_epi_t

        ob = self._build_ob()

        self.sol = torch.zeros(self.g.batch_size, num_samples, device=self.device)
        self.select_edges = torch.zeros(self.g.batch_size, num_samples, device=self.device)

        return ob
