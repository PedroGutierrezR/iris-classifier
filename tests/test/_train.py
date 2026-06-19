import unittest
from src.train import accuracy
from pathlib import Path

class TestTrainAccuracy(unittest.TestCase):

    def test_accuracy(self):
        self.assertGreaterEqual(accuracy, 0.9)

    def test_file(self):
        path = Path(__file__).parent.parent.joinpath("../outputs/confusion/_matrix.png").resolve()
        self.assertTrue(path.is_file())

if __name__ == '__main__':
    unittest.main()
