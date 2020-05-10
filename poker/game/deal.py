import torch


def deal(cards: torch.Tensor, dealt: int = 0):
    batch_size = cards.shape[0]
    size = cards.shape[1]

    for n in range(dealt, size):
        if n >= 1:
            ordered, _ = torch.sort(cards[:, :n], dim=1)

        cards[:, n] = torch.randint(0, 52 - n, (batch_size,), device=cards.device)

        for i in range(n):
            cards[:, n] += cards[:, n] >= ordered[:, i]


def test_deal():
    cards = torch.zeros((100, 7), dtype=torch.long)
    deal(cards, 0)
    print(cards)


if __name__ == '__main__':
    test_deal()
