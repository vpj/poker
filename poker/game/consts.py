SUITS = ['S', 'H', 'D', 'C']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'X', 'J', 'Q', 'K', 'A']
N_SUITS = 4
N_RANKS = 13
N_CARDS = N_SUITS * N_RANKS

assert len(SUITS) == N_SUITS
assert len(RANKS) == N_RANKS


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
SCORE_RANGE = {
    Hands.high_card: 2 ** 13,
    Hands.pair: 13 * (2 ** 13),
    Hands.two_pairs: (13 ** 2) * (2 ** 13),
    Hands.three_of_a_kind: 13 * (2 ** 13),
    Hands.straight: 13,
    Hands.flush: 2 ** 13,
    Hands.full_house: 13 ** 2,
    Hands.four_of_a_kind: 13 ** 2,
    Hands.straight_flush: 13
}


def _calc_offsets():
    offset = 0
    for hand in HANDS_ORDER:
        SCORE_OFFSET[hand] = offset
        offset += SCORE_RANGE[hand]


_calc_offsets()
