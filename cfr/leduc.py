import json
import pathlib
from typing import List, NewType, Dict, cast

import numpy as np
from labml import experiment

from cfr import History as _History, InfoSet as _InfoSet, CFR

Player = NewType('Player', int)
Action = NewType('Action', str)

ONLINE_UPDATE = False
START_ACTIONS = cast(List[Action], ['c', 'r'])
CHECKED_ACTIONS = cast(List[Action], ['c', 'r'])
RAISED_ACTIONS = cast(List[Action], ['f', 'c', 'r'])
RERAISED_ACTIONS = cast(List[Action], ['f', 'c'])

CARDS = ['K', 'Q', 'J'] * 2
PLAYERS = cast(List[Player], [0, 1])

CARD_VALUE = {
    'K': 3,
    'Q': 2,
    'J': 1
}


class InfoSet(_InfoSet):
    def __init__(self, key: str, rnd_history: str):
        self.rnd_history = rnd_history
        super().__init__(key)

    def to_dict(self):
        data = super().to_dict()

        data.update({
            'rnd_history': self.rnd_history,
        })

        return data

    @staticmethod
    def from_dict(data: Dict[str, any]):
        res = InfoSet(data['key'], data['rnd_history'])
        res.load_dict(data)

        return res

    def actions(self) -> List[Action]:
        if self.rnd_history == '':
            return START_ACTIONS
        elif self.rnd_history == 'c':
            return CHECKED_ACTIONS
        elif self.rnd_history[-2:] == 'rr':
            return RERAISED_ACTIONS
        else:
            return RAISED_ACTIONS

    def __repr__(self):
        total = sum(r for r in self.average_strategy.values())
        total = max(total, 1e-6)

        return ' '.join(
            [f'{100 * self.average_strategy.get(a, 0) / total: .1f}%' for a in self.actions()])


class BettingRound:
    history: str

    def __init__(self, history: str):
        self.history = history

    def is_fold(self):
        if len(self.history) >= 1 and self.history[-1] == 'f':
            return True
        else:
            return False

    def is_complete(self):
        if self.history == 'c':
            return False
        if len(self.history) > 0 and self.history[-1] in ['c', 'f']:
            return True

        return False

    def pot(self):
        p = 0
        raised = False
        for s in self.history:
            if not raised and s == 'r':
                raised = True
            elif raised and s in ['r', 'c']:
                p += 1

        return p

    def last_player(self):
        return 1 - (len(self.history) % 2)

    def __add__(self, other: Action):
        assert not self.is_complete()
        return BettingRound(self.history + other)

    def __repr__(self):
        return self.history

    def pretty(self, length=4):
        return self.history + '_' * (length - len(self.history))


class History(_History):
    board_card: str
    player_cards: str
    round1: BettingRound
    round2: BettingRound
    state: str

    def __init__(self, board_card: str, player_cards: str, round1: BettingRound,
                 round2: BettingRound, state: str):
        self.state = state
        self.round2 = round2
        self.round1 = round1
        self.player_cards = player_cards
        self.board_card = board_card

    def is_terminal(self):
        if self.state == 'round1' and self.round1.is_fold():
            return True
        elif self.state == 'round2' and self.round2.is_complete():
            return True
        else:
            return False

    def pot(self):
        return 1 + self.round1.pot() * 2 + self.round2.pot() * 4

    def terminal_utility(self, i: Player) -> float:
        pot = self.pot()
        if self.round1.is_fold():
            winner = 1 - self.round1.last_player()
        elif self.round1.is_fold():
            winner = 1 - self.round2.last_player()
        else:
            if self.board_card == self.player_cards[0]:
                winner = 0
            elif self.board_card == self.player_cards[1]:
                winner = 1
            elif CARD_VALUE[self.player_cards[0]] > CARD_VALUE[self.player_cards[1]]:
                winner = 0
            elif CARD_VALUE[self.player_cards[0]] < CARD_VALUE[self.player_cards[1]]:
                winner = 1
            else:
                return 0

        if i == winner:
            return pot
        else:
            return -pot

    def is_chance(self) -> bool:
        return self.state in ['deal', 'open']

    def __add__(self, other: Action):
        state = self.state
        round1 = self.round1
        round2 = self.round2
        if self.state == 'deal':
            assert other == ''
            state = 'round1'
        elif self.state == 'open':
            assert other == ''
            state = 'round2'
        elif self.state == 'round1':
            round1 = self.round1 + other
            if round1.is_complete() and not round1.is_fold():
                state = 'open'
        elif self.state == 'round2':
            round2 = self.round2 + other
        else:
            assert False

        return History(self.board_card, self.player_cards, round1, round2, state)

    def info_set_key(self) -> str:
        i = self.player()
        h = self.player_cards[i]
        if self.state == 'round2':
            h += self.board_card
        else:
            h += '_'
        h += f'.{self.round1.pretty()}'
        h += f'.{self.round2.pretty()}'

        return h

    def new_info_set(self) -> InfoSet:
        if self.state == 'round1':
            rnd = self.round1
        else:
            rnd = self.round2

        h = self.info_set_key()
        return InfoSet(h, rnd.history)

    def player(self) -> Player:
        if self.state == 'round1':
            return cast(Player, 1 - self.round1.last_player())
        elif self.state == 'round2':
            return cast(Player, 1 - self.round2.last_player())
        else:
            assert False

    def sample_chance(self) -> Action:
        return cast(Action, '')

    def __repr__(self):
        return f'{self.player_cards} {self.board_card} {self.state} {self.round1} {self.round2}'


def create_new_history():
    used = set()
    cards = ''
    while len(cards) < 3:
        r = np.random.randint(len(CARDS))
        card = CARDS[r]
        if r in used:
            continue
        used.add(r)
        cards += card

    return History(cards[0], cards[1:], BettingRound(''), BettingRound(''), 'deal')


class InfoSetSaver(experiment.ModelSaver):
    def __init__(self, infosets: Dict[str, InfoSet]):
        self.infosets = infosets

    def save(self, checkpoint_path: pathlib.Path) -> any:
        data = {key: infoset.to_dict() for key, infoset in self.infosets.items()}
        file_name = f"infosets.json"

        with open(str(checkpoint_path / file_name), 'w') as f:
            f.write(json.dumps(data))

        return file_name

    def load(self, checkpoint_path: pathlib.Path, file_name: str):
        with open(str(checkpoint_path / file_name), 'w') as f:
            data = json.loads(f.read())

        for key, d in data.items():
            self.infosets[key] = InfoSet.from_dict(d)


if __name__ == '__main__':
    experiment.create(name='leduc_poker', writers={'sqlite', 'web_api'})
    cfr = CFR(create_new_history=create_new_history,
              epochs=2_000_000,
              track_frequency=100,
              save_frequency=1_000)
    experiment.add_model_savers({'info_sets': InfoSetSaver(cfr.info_sets)})
    experiment.start()
    cfr.solve()
