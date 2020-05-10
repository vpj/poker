import torch
from lab import tracker, experiment
from lab.helpers.pytorch.module import Module
from torch import nn

from poker.game.consts import N_CARDS
from poker.game.deal import deal
from poker.game.scorer import score
from who_won import Configs as WhoWonConfigs


class Probabilities(Module):
    def __init__(self, *, size=1024, layers=6):
        super().__init__()

        self.card_embedding = nn.Embedding(N_CARDS, size)
        self.type_embedding = nn.Embedding(3, size)
        self.gate_sigmoid = nn.Sigmoid()

        layers = [nn.Linear(size, size) for _ in range(layers)]
        self.layers = nn.ModuleList(layers)

        self.activation = nn.ReLU()
        self.final = nn.Linear(size, 3)
        self.card_type = nn.Parameter(torch.tensor([0, 0, 2, 2, 2, 2, 2, 1, 1]),
                                      requires_grad=False)
        self.masks = nn.Parameter(torch.zeros((10, 9), dtype=torch.bool),
                                  requires_grad=False)
        for i in range(10):
            self.masks[i, 0:i] = True

    def __call__(self, cards: torch.Tensor):
        n_cards = cards.shape[1]

        batch_size = cards.shape[0]
        embeddings = self.card_embedding(cards.view(-1)).view(batch_size, n_cards, -1)
        gates = self.gate_sigmoid(self.type_embedding(self.card_type[:n_cards]))
        gates = gates.view(1, n_cards, -1)

        x = embeddings * gates
        x = x.sum(dim=1)

        for layer in self.layers:
            x = self.activation(layer(x)) + x

        return self.final(x)


class Configs(WhoWonConfigs):
    labels_batch: torch.Tensor = 'win_lose_labels'
    loss_func = nn.KLDivLoss()
    model: Probabilities = 'probabilities_model'
    n_cards: int = 2
    samples_size: int = 5_000
    batch_size: int = 128
    validate_loss = nn.MSELoss()

    def train(self):
        start = torch.zeros((self.batch_size, self.n_cards), dtype=torch.long, device=self.device)
        deal(start)
        rep = start.view(-1, 1, self.n_cards)
        rep = rep.repeat(1, self.samples_size, 1)
        cards = start.new_zeros(self.batch_size, self.samples_size, 9)
        cards[:, :, :self.n_cards] = rep
        cards = cards.view(-1, 9)
        deal(cards, self.n_cards)
        score0 = score(cards[:, :7]).view(self.batch_size, -1)
        score1 = score(cards[:, -7:]).view(self.batch_size, -1)

        labels = cards.new_zeros((self.batch_size, 3), dtype=torch.float)
        labels[:, 0] = (score0 > score1).to(torch.float).mean(-1)
        labels[:, 1] = (score0 == score1).to(torch.float).mean(-1)
        labels[:, 2] = (score0 < score1).to(torch.float).mean(-1)

        pred = torch.log_softmax(self.model(start), dim=-1)
        loss = self.loss_func(pred, labels)
        tracker.add('train.loss', loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        tracker.set_queue('train.loss', is_print=True)
        for s in self.training_loop:
            self.train()


@Configs.calc(Configs.model)
def probabilities_model(c: Configs):
    return Probabilities().to(c.device)


def main():
    conf = Configs()
    experiment.create(name='probabilities_fixed_cards')
    experiment.calculate_configs(conf,
                                 {},
                                 ['run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
