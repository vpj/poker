import torch
from lab import monit

from poker.consts import N_RANKS, SCORE_OFFSET, Hands, SCORE_RANGE

STRAIGHT_MASKS = [(2 ** 4) - 1 + (2 ** 12)]


def _calc_straight_masks():
    for i in range(13 - 5):
        STRAIGHT_MASKS.append((2 ** i) * ((2 ** 5) - 1))


_calc_straight_masks()


class Sequences:
    def __init__(self, cards: torch.Tensor):
        self.suit_one_hot = torch.eye(4, device=cards.device, dtype=cards.dtype)
        self.rank_one_hot = torch.eye(13, device=cards.device, dtype=cards.dtype)
        self.rank_value = torch.arange(13,
                                       device=cards.device,
                                       dtype=cards.dtype).unsqueeze(0)
        self.straight_masks = torch.tensor(STRAIGHT_MASKS,
                                           device=cards.device,
                                           dtype=cards.dtype).unsqueeze(0)
        self.straight_value = torch.arange(13 - 5 + 1,
                                           device=cards.device,
                                           dtype=cards.dtype).unsqueeze(0)
        self.batch_size = cards.shape[0]
        self.ranks = (cards % N_RANKS)
        self.suits = (cards / N_RANKS)

    def _straight_score(self, ranks: torch.Tensor):
        one_hot = self.rank_one_hot[ranks.view(-1)].view(-1, 13)
        value = (one_hot.sum(dim=0) > 0) * self.rank_value
        mask = (2 ** value).sum(dim=1).unsqueeze(1)
        straights = (mask & self.straight_masks[0]) == self.straight_masks[0]
        straight_values = straights * (self.straight_value[0] + 1)
        return straight_values.max() - 1

    def _flush_score(self, ranks: torch.Tensor):
        ranks, _ = torch.sort(ranks, dim=-1, descending=True)
        ranks = ranks[:5]
        return (2 ** ranks).sum()

    def calc_score_dumb(self, ranks: torch.Tensor, suits: torch.Tensor):
        suit_one_hot = self.suit_one_hot[suits.view(-1)].view(-1, 4)
        suit_count = suit_one_hot.sum(dim=0)

        mx, arg_max = suit_count.max(-1)
        if mx >= 5:
            flush_ranks = ranks.masked_select(suits == arg_max)
            straight_score = self._straight_score(flush_ranks)
            if straight_score >= 0:
                assert 0 <= straight_score < SCORE_RANGE[Hands.straight_flush]
                return SCORE_OFFSET[Hands.straight_flush] + straight_score
            flush_score = self._flush_score(flush_ranks)
            assert 0 <= flush_score < SCORE_RANGE[Hands.flush]
            return SCORE_OFFSET[Hands.flush] + flush_score

        straight_score = self._straight_score(ranks)
        if straight_score >= 0:
            assert 0 <= straight_score < SCORE_RANGE[Hands.straight]
            return SCORE_OFFSET[Hands.straight] + straight_score
        else:
            return 0

    def __call__(self, is_dumb=False):
        if is_dumb:
            scores = self.ranks.new_zeros(self.batch_size)
            for i in range(self.batch_size):
                scores[i] = self.calc_score_dumb(self.ranks[i], self.suits[i])

            return scores
        else:
            return self.calc_score()

    def calc_score(self):
        scores = self.ranks.new_zeros(self.batch_size)

        rank_one_hot = self.rank_one_hot[self.ranks.view(-1)].view(self.batch_size, -1, 13)

        suit_one_hot = self.suit_one_hot[self.suits.view(-1)].view(self.batch_size, -1, 4)
        suit_count = suit_one_hot.sum(dim=1)

        mx, arg_mx = suit_count.max(-1)
        arg_mx = arg_mx.unsqueeze(-1)
        flush_rank_one_hot = rank_one_hot * (arg_mx == self.suits).unsqueeze(-1)
        flush_rank_mask = flush_rank_one_hot.sum(dim=1) > 0
        flush_rank_value = flush_rank_mask * self.rank_value
        flush_rank_mask = ((2 ** flush_rank_value) * flush_rank_mask).sum(dim=1)

        straight_flush = (flush_rank_mask.unsqueeze(
            -1) & self.straight_masks) == self.straight_masks
        straight_flush_values = straight_flush * (self.straight_value + 1)
        straight_flush_score, _ = straight_flush_values.max(-1)
        straight_flush = straight_flush_score > 0
        scores += straight_flush * SCORE_OFFSET[Hands.straight_flush]
        scores += straight_flush * (straight_flush_score - 1)

        for i in range(2):
            is_remove = (mx > 5).to(mx.dtype)
            flush_rank_mask = flush_rank_mask & (flush_rank_mask - is_remove)
            mx -= is_remove

        flush = (mx == 5) & ~straight_flush
        scores += flush * SCORE_OFFSET[Hands.flush]
        scores += flush_rank_mask * flush

        rank_mask = rank_one_hot.sum(dim=1) > 0
        rank_value = rank_mask * self.rank_value
        rank_mask = (2 ** rank_value).sum(dim=1)
        straights = (rank_mask.unsqueeze(-1) & self.straight_masks) == self.straight_masks
        straight_values = straights * (self.straight_value + 1)
        straight_scores, _ = straight_values.max(-1)
        straight = (straight_scores > 0) & ~flush
        scores += straight * SCORE_OFFSET[Hands.straight]
        scores += straight * (straight_scores - 1)

        return scores


def _test():
    from poker.deal import deal

    cards = torch.zeros((20_000, 7), dtype=torch.long)
    deal(cards, 0)
    # cards[0] = torch.tensor([8, 30, 33, 35, 36, 37, 39])
    scorer = Sequences(cards)
    with monit.section("Dumb"):
        scores = scorer(is_dumb=True)
    with monit.section("Vector"):
        scores_vec = scorer()

    print((scores != scores_vec).sum())
    print('scores')


if __name__ == '__main__':
    _test()
