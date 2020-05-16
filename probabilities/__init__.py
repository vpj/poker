import torch
from labml import tracker, experiment
from labml.helpers.pytorch.module import Module
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
        assert cards.shape[1] == 9

        batch_size = cards.shape[0]
        embeddings = self.card_embedding(cards.view(-1)).view(batch_size, 9, -1)
        gates = self.gate_sigmoid(self.type_embedding(self.card_type))
        gates = gates.view(1, 9, -1)

        x = embeddings * gates
        x = x.cumsum(dim=1)

        for layer in self.layers:
            x = self.activation(layer(x)) + x

        return self.final(x)


class Configs(WhoWonConfigs):
    loss_func = nn.CrossEntropyLoss()
    model: Probabilities = 'probabilities_model'
    valid_internal: int = 100_000
    validate_samples_size: int = 10_000
    validate_batch_size: int = 64
    validate_cards_batch: torch.Tensor
    validate_length: int = 2
    validate_loss = nn.MSELoss()

    def train(self):
        deal(self.cards_batch)
        score0 = score(self.cards_batch[:, :7])
        score1 = score(self.cards_batch[:, -7:])
        self.labels_batch.zero_()
        self.labels_batch += (score0 == score1) * 1
        self.labels_batch += (score0 < score1) * 2

        pred = self.model(self.cards_batch)
        loss = self.loss_func(pred.view(-1, 3),
                              self.labels_batch.view(-1, 1).repeat(1, 9).view(-1))
        tracker.add('train.loss', loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def validate(self):
        start = self.validate_cards_batch.new_zeros((self.validate_batch_size, 9))
        deal(start)
        rep = start[:, :self.validate_length].view(-1, 1, self.validate_length)
        rep = rep.repeat(1, self.validate_samples_size, 1)
        self.validate_cards_batch[:, :self.validate_length] = rep.view(-1, self.validate_length)
        deal(self.validate_cards_batch, self.validate_length)
        score0 = score(self.validate_cards_batch[:, :7]).view(self.validate_batch_size, -1)
        score1 = score(self.validate_cards_batch[:, -7:]).view(self.validate_batch_size, -1)
        labels = self.validate_cards_batch.new_zeros((self.validate_batch_size, 3),
                                                     dtype=torch.float)
        labels[:, 0] = (score0 > score1).to(torch.float).mean(-1)
        labels[:, 1] = (score0 == score1).to(torch.float).mean(-1)
        labels[:, 2] = (score0 < score1).to(torch.float).mean(-1)

        pred = torch.softmax(self.model(start)[:, self.validate_length, :], dim=-1)
        loss = self.validate_loss(pred, labels)
        tracker.add('valid.loss', loss)

    def run(self):
        tracker.set_queue('train.loss', is_print=True)
        tracker.set_scalar('valid.loss', is_print=True)
        for s in self.training_loop:
            self.train()
            if self.training_loop.is_interval(self.valid_internal):
                self.validate()


@Configs.calc(Configs.validate_cards_batch)
def allocate_validate_batch(c: Configs):
    return torch.zeros((c.validate_samples_size * c.validate_batch_size, 9),
                       dtype=torch.long, device=c.device)


@Configs.calc(Configs.model)
def probabilities_model(c: Configs):
    return Probabilities().to(c.device)


def main():
    conf = Configs()
    experiment.create(name='probabilities')
    experiment.calculate_configs(conf,
                                 {},
                                 ['run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
