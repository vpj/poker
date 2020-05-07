import torch

from poker.consts import N_RANKS, SCORE_OFFSET, Hands, SCORE_RANGE
from lab import logger, monit

MASK = 2 ** 13

class Combinations:
    def __init__(self, cards: torch.Tensor):
        self.suit_one_hot = torch.eye(4, device=cards.device, dtype=cards.dtype)
        self.rank_one_hot = torch.eye(13, device=cards.device, dtype=cards.dtype)
        self.rank_value = torch.arange(13, device=cards.device, dtype=cards.dtype).unsqueeze(0)
        self.batch_size = cards.shape[0]
        self.ranks = (cards % N_RANKS)

    def calc_score_dumb(self, ranks: torch.Tensor):
        rank_one_hot = self.rank_one_hot[ranks].view(-1, 13)
        rank_count = rank_one_hot.sum(dim=0)
        rank_value = self.rank_value[0]
        count_score = rank_count * 16 + rank_value * (rank_count != 0)

        count_score, _ = torch.sort(count_score, dim=-1, descending=True)
        hand = count_score % 16
        if count_score[0] >= 4 * 16:
            next_card = torch.max(hand[1:])
            score = (hand[0] * 13) + next_card
            assert 0 <= score < SCORE_RANGE[Hands.four_of_a_kind]
            return SCORE_OFFSET[Hands.four_of_a_kind] + score

        if count_score[0] >= 3 * 16 and count_score[1] >= 2 * 16:
            score = (hand[0] * 13) + hand[1]
            assert 0 <= score < SCORE_RANGE[Hands.full_house]
            return SCORE_OFFSET[Hands.full_house] + score

        if count_score[0] >= 3 * 16:
            score = (hand[0] * MASK) + (2 ** hand[1]) + (2 ** hand[2])
            assert 0 <= score < SCORE_RANGE[Hands.three_of_a_kind]
            return SCORE_OFFSET[Hands.three_of_a_kind] + score

        if count_score[0] >= 2 * 16 and count_score[1] >= 2 * 16:
            next_card = torch.max(hand[2:])
            score = ((hand[0] * 13 + hand[1]) * MASK) + (2 ** next_card)
            assert 0 <= score < SCORE_RANGE[Hands.two_pairs]
            return SCORE_OFFSET[Hands.two_pairs] + score

        if count_score[0] >= 2 * 16:
            score = (hand[0] * MASK) + (2 ** hand[1]) + (2 ** hand[2]) + (2 ** hand[3])
            assert 0 <= score < SCORE_RANGE[Hands.pair]
            return SCORE_OFFSET[Hands.pair] + score

        score = (2 ** hand[0]) + (2 ** hand[1]) + (2 ** hand[2]) + (2 ** hand[3]) + (2 ** hand[4])
        assert 0 <= score < SCORE_RANGE[Hands.high_card]
        return SCORE_OFFSET[Hands.high_card] + score

    def __call__(self, is_dumb=False):
        if is_dumb:
            scores = self.ranks.new_zeros(self.batch_size)
            for i in range(self.batch_size):
                scores[i] = self.calc_score_dumb(self.ranks[i])

            return scores
        else:
            return self.calc_rank_comb_score()

    def calc_rank_comb_score(self):
        rank_one_hot = self.rank_one_hot[self.ranks.view(-1)].view(self.batch_size, -1, 13)
        rank_count = rank_one_hot.sum(dim=1)

        scores = self.ranks.new_zeros(self.batch_size)
        collected = self.ranks.new_zeros(self.batch_size)
        high_cards = self.ranks.new_zeros(self.batch_size)

        with monit.section("First combination"):
            count_score = rank_count * 16 + self.rank_value * (rank_count != 0)
            mx, arg_mx1 = count_score.max(-1)
            rank_count.scatter_(-1, arg_mx1.unsqueeze(-1), 0)

            four_of_a_kind = mx >= (4 * 16)
            mx = mx * ~four_of_a_kind
            rank_count = ((rank_count * ~four_of_a_kind.view(-1, 1)) +
                          (rank_count.clamp_max(1) * four_of_a_kind.view(-1, 1)))

            three_of_a_kind = mx >= (3 * 16)
            mx = mx * ~three_of_a_kind

            pair = mx >= (2 * 16)

        with monit.section("Second combination"):
            count_score = rank_count * 16 + self.rank_value * (rank_count != 0)
            mx, arg_mx2 = count_score.max(-1)
            rank_count.scatter_(-1, arg_mx2.unsqueeze(-1), 0)

            second_pair = mx > (2 * 16)

        with monit.section("Calculate scores"):
            scores += four_of_a_kind * SCORE_OFFSET[Hands.four_of_a_kind]
            scores += four_of_a_kind * ((arg_mx1 * 13) + arg_mx2)
            collected += four_of_a_kind * 5

            full_house = three_of_a_kind & second_pair
            scores += full_house * SCORE_OFFSET[Hands.full_house]
            scores += full_house * ((arg_mx1 * 13) + arg_mx2)
            collected += full_house * 5
            assert (four_of_a_kind & full_house).sum() == 0

            three_of_a_kind = three_of_a_kind & ~second_pair
            scores += three_of_a_kind * SCORE_OFFSET[Hands.three_of_a_kind]
            scores += three_of_a_kind * (arg_mx1 * MASK)
            high_cards += three_of_a_kind * (2 ** arg_mx2)
            collected += three_of_a_kind * 4
            assert ((four_of_a_kind | full_house) & three_of_a_kind).sum() == 0

            two_pairs = pair & second_pair
            scores += two_pairs * SCORE_OFFSET[Hands.two_pairs]
            scores += two_pairs * ((arg_mx1 * 13 + arg_mx2) * MASK)
            collected += two_pairs * 4
            assert ((four_of_a_kind | full_house | three_of_a_kind) & two_pairs).sum() == 0

            pair = pair & ~second_pair
            scores += pair * SCORE_OFFSET[Hands.pair]
            scores += pair * (arg_mx1 * MASK)
            high_cards += pair * (2 ** arg_mx2)
            collected += pair * 3
            assert ((four_of_a_kind | full_house | three_of_a_kind | two_pairs) & pair).sum() == 0

            high_card = ~(four_of_a_kind | full_house | three_of_a_kind | two_pairs | pair)
            scores += high_card * SCORE_OFFSET[Hands.high_card]
            high_cards += high_card * ((2 ** arg_mx1) + (2 ** arg_mx2))
            collected += pair * 2

        for i in range(2, 5):
            is_collect = collected < 5
            count_score = rank_count * 16 + self.rank_value * (rank_count != 0)
            mx, arg_mx = count_score.max(-1)
            rank_count.scatter_(-1, arg_mx.unsqueeze(-1), 0)

            high_cards += is_collect * (2 ** arg_mx)
            collected += is_collect

        scores += high_cards

        return scores


def _test():
    from poker.deal import deal

    cards = torch.zeros((1000, 7), dtype=torch.long)
    deal(cards, 0)
    # cards[0] = torch.tensor([3,  6, 16, 17, 27, 44, 45])
    scorer = Combinations(cards)
    with monit.section("Dumb"):
        scores = scorer(is_dumb=True)
    with monit.section("Vector"):
        scores_vec = scorer()

    print(scores)


if __name__ == '__main__':
    _test()
