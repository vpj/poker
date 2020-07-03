from typing import NewType, Dict, List, Callable, cast

from labml import monit, tracker, logger, experiment

Action = NewType('Action', str)
Player = NewType('Player', int)


class InfoSet:
    key: str
    regret: Dict[Action, float]
    current_regret: Dict[Action, float]
    average_strategy: Dict[Action, float]
    strategy: Dict[Action, float]

    def __init__(self, key: str):
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.current_regret = {a: 0 for a in self.actions()}
        self.average_strategy = {a: 0 for a in self.actions()}
        self.calculate_policy()

    def actions(self) -> List[Action]:
        raise NotImplementedError()

    def to_dict(self):
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.average_strategy,
        }

    def load_dict(self, data: Dict[str, any]):
        self.regret = data['regret']
        self.average_strategy = data['average_strategy']
        self.calculate_policy()

    def calculate_policy(self):
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(r for r in regret.values())
        if regret_sum > 0:
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        else:
            count = len(list(a for a in self.regret))
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def clear(self):
        self.current_regret = {a: 0 for a in self.actions()}

    def update_regrets(self):
        for k, v in self.current_regret.items():
            self.regret[k] += v

    def __repr__(self):
        raise NotImplementedError()


class History:
    def is_terminal(self):
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        raise NotImplementedError()

    def is_chance(self) -> bool:
        raise NotImplementedError()

    def __add__(self, action: Action):
        raise NotImplementedError()

    def info_set_key(self) -> str:
        raise NotImplementedError

    def new_info_set(self) -> InfoSet:
        raise NotImplementedError()

    def player(self) -> int:
        raise NotImplementedError()

    def sample_chance(self) -> Action:
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class CFR:
    track_frequency: int
    info_sets: Dict[str, InfoSet]
    is_online_update: bool
    n_players: int
    create_new_history: Callable[[], History]
    epochs: int

    def __init__(self, *, create_new_history, epochs,
                 is_online_update=False, n_players=2,
                 track_frequency=10,
                 save_frequency=10):
        self.save_frequency = save_frequency
        self.track_frequency = track_frequency
        self.n_players = n_players
        self.is_online_update = is_online_update
        self.epochs = epochs
        self.create_new_history = create_new_history
        self.info_sets = {}

        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')
        tracker.set_histogram(f'current_regret.*')

    def cfr(self, h: History, i: Player, pi: List[float]) -> float:
        if h.is_terminal():
            return h.terminal_utility(i)
        elif h.is_chance():
            a = h.sample_chance()
            return self.cfr(h + a, i, pi)

        info_set_key = h.info_set_key()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        I = self.info_sets[info_set_key]
        v = 0
        va = {}

        pi_neg_i = 1
        for j, pi_j in enumerate(pi):
            if j != h.player():
                pi_neg_i *= pi_j

        for a in I.actions():
            pi_next = pi.copy()
            pi_next[h.player()] *= I.strategy[a]
            va[a] = self.cfr(h + a, i, pi_next)
            v = v + I.strategy[a] * va[a]

        if h.player() == i:
            for a in I.actions():
                if self.is_online_update:
                    I.regret[a] += pi_neg_i * (va[a] - v)
                else:
                    I.current_regret[a] += pi_neg_i * (va[a] - v)
                I.average_strategy[a] = I.average_strategy[a] + pi[i] * I.strategy[a]

        if self.is_online_update:
            I.calculate_policy()

        return v

    def update(self):
        for k, I in self.info_sets.items():
            I.update_regrets()
            I.calculate_policy()

    def solve(self):
        for t in monit.loop(self.epochs):
            if not self.is_online_update:
                for I in self.info_sets.values():
                    I.clear()
            for i in range(self.n_players):
                self.cfr(self.create_new_history(), cast(Player, i),
                         [1 for _ in range(self.n_players)])
            if not self.is_online_update:
                self.update()
            with monit.section("Track"):
                for I in self.info_sets.values():
                    for a in I.actions():
                        tracker.add({
                            f'strategy.{I.key}.{a}': I.strategy[a],
                            f'average_strategy.{I.key}.{a}': I.average_strategy[a],
                            f'regret.{I.key}.{a}': I.regret[a],
                            f'current_regret.{I.key}.{a}': I.current_regret[a]
                        })

            if t % self.track_frequency == 0:
                tracker.save()
                logger.log()

            if (t + 1) % self.save_frequency == 0:
                experiment.save_checkpoint()

        logger.inspect(self.info_sets)
