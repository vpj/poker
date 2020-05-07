import torch

SUITS = ['S', 'H', 'D', 'C']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'X', 'J', "K', 'A"]

assert len(SUITS) == 4
assert len(RANKS) == 13

SUIT_ONE_HOT = torch.eye(4)
RANK_ONE_HOT = torch.eye(13)
RANK_VALUE = torch.arange(13)

MASK = 1 << 13


class Hands:
    high_card = 'high_card'
    pair = 'pair'
    two_pairs = 'two_pairs'
    three_of_a_kind = 'three_of_a_kind'
    straight = 'straight'
    flush = 'flush'
    full_house = 'full_house'
    four_of_a_kind = 'four_of_a_kind'
    straight_flush = 'straight_flush'


HANDS_ORDER = [
    Hands.high_card,
    Hands.pair,
    Hands.two_pairs,
    Hands.three_of_a_kind,
    Hands.straight,
    Hands.flush,
    Hands.full_house,
    Hands.four_of_a_kind,
    Hands.straight_flush
]

SCORE_OFFSET = {}


def calc_rank_comb_score(cards: torch.Tensor):
    ranks = cards & 15
    batch_size = ranks.shape[0]
    ranks.view(-1)
    rank_one_hot = RANK_ONE_HOT[ranks].view(batch_size, -1, 13)
    rank_count = rank_one_hot.sum(dim=1)
    rank_value = RANK_VALUE.unsqueeze(0)

    scores = rank_count.new_zeros(batch_size)
    collected = rank_count.new_zeros(batch_size)
    high_cards = rank_count.new_zeros(batch_size)

    # First
    count_score = (rank_count - 1) * 16 + rank_value * (rank_count != 0)
    mx, arg_mx = count_score.max(-1)
    rank_count.scatter_(-1, arg_mx.unsqueeze(-1), 0)

    four_of_a_kind = mx > (4 * 16)
    mx = mx * ~four_of_a_kind
    rank_count = ((rank_count * ~four_of_a_kind.view(-1, 1)) +
                  (rank_count.clamp_max(1) * four_of_a_kind.view(-1, 1)))

    three_of_a_kind = mx > (3 * 16)
    mx = mx * three_of_a_kind

    pair = mx > (2 * 16)

    # Second
    count_score = (rank_count - 1) * 16 + rank_value * (rank_count != 0)
    mx, arg_mx2 = count_score.max(-1)
    rank_count.scatter_(-1, arg_mx2.unsqueeze(-1), 0)

    second_pair = mx > (2 * 16)

    # Calculate scores
    scores += four_of_a_kind * SCORE_OFFSET[Hands.four_of_a_kind]
    scores += four_of_a_kind * ((arg_mx * 13) + arg_mx)
    collected += four_of_a_kind * 5

    full_house = three_of_a_kind & second_pair
    scores += full_house * SCORE_OFFSET[Hands.full_house]
    scores += full_house * ((arg_mx * 13) + arg_mx2)
    collected += full_house * 5
    assert (four_of_a_kind & full_house).sum() == 0

    three_of_a_kind = three_of_a_kind & ~second_pair
    scores += three_of_a_kind * SCORE_OFFSET[Hands.three_of_a_kind]
    scores += three_of_a_kind * (arg_mx << 13)
    high_cards += three_of_a_kind * (1 << arg_mx2)
    collected += three_of_a_kind * 4
    assert ((four_of_a_kind | full_house) & three_of_a_kind).sum() == 0

    two_pairs = pair & second_pair
    scores += two_pairs * SCORE_OFFSET[Hands.two_pairs]
    scores += two_pairs * ((arg_mx * 13 + arg_mx2) << 13)
    collected += two_pairs * 4
    assert ((four_of_a_kind | full_house | three_of_a_kind) & two_pairs).sum() == 0

    pair = pair & ~second_pair
    scores += pair * SCORE_OFFSET[Hands.pair]
    scores += pair * (arg_mx << 13)
    high_cards += pair * (1 << arg_mx2)
    collected += pair * 3
    assert ((four_of_a_kind | full_house | three_of_a_kind | two_pairs) & pair).sum() == 0

    high_card = ~(four_of_a_kind | full_house | three_of_a_kind | two_pairs | pair)
    scores += high_card * SCORE_OFFSET[Hands.high_card]
    high_cards += high_card * ((1 << arg_mx) + (1 << arg_mx2))
    collected += pair * 2

    for i in range(2, 5):
        is_collect = collected < 5
        count_score = (rank_count - 1) * 16 + rank_value * (rank_count != 0)
        mx, arg_mx = count_score.max(-1)
        rank_count.scatter_(-1, arg_mx.unsqueeze(-1), 0)

        high_cards += is_collect * (1 << arg_mx)

    scores += high_cards

    return scores


def calc_order_score(cards: torch.Tensor):
    pass
