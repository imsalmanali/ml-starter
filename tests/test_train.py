from src.train import train


def test_train_returns_reasonable_accuracy():
    accuracy = train()
    assert accuracy >= 0.8
