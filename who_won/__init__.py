import torch
from lab import tracker, experiment
from lab.helpers.pytorch.device import DeviceConfigs
from lab.helpers.pytorch.module import Module
from lab.helpers.training_loop import TrainingLoopConfigs
from torch import nn

from poker.game.consts import N_CARDS
from poker.game.deal import deal
from poker.game.scorer import score


class WhoWon(Module):
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

    def __call__(self, cards: torch.Tensor):
        assert cards.shape[1] == 9
        batch_size = cards.shape[0]
        embeddings = self.card_embedding(cards.view(-1)).view(batch_size, 9, -1)
        gates = self.gate_sigmoid(self.type_embedding(self.card_type))
        gates = gates.view(1, 9, -1)

        x = embeddings * gates
        x = x.sum(dim=1)

        for layer in self.layers:
            x = self.activation(layer(x)) + x

        return self.final(x)


class SimpleAccuracy:
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> int:
        pred = output.argmax(dim=1)
        return pred.eq(target).sum().item()


class Configs(DeviceConfigs, TrainingLoopConfigs):
    batch_size: int = 1024
    cards_batch: torch.Tensor
    labels_batch: torch.Tensor
    loss_func = nn.CrossEntropyLoss()
    accuracy_func = SimpleAccuracy()
    optimizer: torch.optim.Adam = 'adam_optimizer'
    model: Module

    loop_step: int = 'loop_step'
    loop_count: int = 10_000_000
    log_new_line_interval = 100_000
    log_write_interval = 1
    save_models_interval = 5000_000

    def train(self):
        deal(self.cards_batch)
        score0 = score(self.cards_batch[:, :7])
        score1 = score(self.cards_batch[:, -7:])
        self.labels_batch.zero_()
        self.labels_batch += (score0 < score1) * 1
        self.labels_batch += (score0 == score1) * 2

        pred = self.model(self.cards_batch)
        loss = self.loss_func(pred, self.labels_batch)
        tracker.add(loss=loss)
        tracker.add(accuracy=self.accuracy_func(pred, self.labels_batch) / self.batch_size)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        tracker.set_queue('loss', is_print=True)
        tracker.set_queue('accuracy', is_print=True)
        for s in self.training_loop:
            self.train()


@Configs.calc(Configs.optimizer)
def adam_optimizer(c: Configs):
    return torch.optim.Adam(c.model.parameters(), lr=1e-3)


@Configs.calc(Configs.loop_step)
def loop_step(c: Configs):
    return c.batch_size


@Configs.calc(Configs.cards_batch)
def allocate_batch(c: Configs):
    return torch.zeros((c.batch_size, 9), dtype=torch.long, device=c.device)


@Configs.calc(Configs.labels_batch)
def allocate_labels(c: Configs):
    return torch.zeros(c.batch_size, dtype=torch.long, device=c.device)


@Configs.calc(Configs.model)
def model(c: Configs):
    return WhoWon().to(c.device)


def main():
    conf = Configs()
    experiment.create(name='who_won')
    experiment.calculate_configs(conf,
                                 {},
                                 ['run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
