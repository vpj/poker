import torch
from lab import monit

from poker.game.combinations import Combinations
from poker.game.sequences import Sequences


class Scorer:
    def __init__(self, cards: torch.Tensor):
        assert cards.shape[1] == 7
        self.combinations = Combinations(cards)
        self.sequences = Sequences(cards)

    def __call__(self):
        return torch.max(self.combinations(),
                         self.sequences())


def score(cards: torch.Tensor):
    return Scorer(cards)()


def _test_gpu():
    from poker.game.deal import deal

    with monit.section("Allocate"):
        cards = torch.zeros((1_000_000, 7), dtype=torch.long, device=torch.device('cuda'))
    with monit.section('Deal'):
        deal(cards, 0)
    scorer = Scorer(cards)
    with monit.section("Score"):
        scores_vec = scorer()


if __name__ == '__main__':
    _test_gpu()
