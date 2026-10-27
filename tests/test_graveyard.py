import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graveyard import bury_rows


def row(i, ok, said):
    return {"id": i, "answer": "1", "right": ok, "extracted": said, "completion_tokens": 10}


class Bury(unittest.TestCase):
    def setUp(self):
        # q1 survives, q2 dies at B and stays dead, q3 dies at C and is right again at D,
        # q4 is wrong at the reference and right at C (a ghost).
        self.by = {
            "A": {"q1": row("q1", True, 1), "q2": row("q2", True, 1), "q3": row("q3", True, 1), "q4": row("q4", False, 2)},
            "B": {"q1": row("q1", True, 1), "q2": row("q2", False, 3), "q3": row("q3", True, 1), "q4": row("q4", False, 2)},
            "C": {"q1": row("q1", True, 1), "q2": row("q2", False, 3), "q3": row("q3", False, 5), "q4": row("q4", True, 1)},
            "D": {"q1": row("q1", True, 1), "q2": row("q2", False, 4), "q3": row("q3", True, 1), "q4": row("q4", False, 2)},
        }
        self.g = bury_rows(self.by, ["A", "B", "C", "D"])

    def test_table(self):
        t = {r["rung"]: r for r in self.g["table"]}
        self.assertEqual([t[r]["right"] for r in "ABCD"], [3, 2, 2, 2])
        self.assertEqual([t[r]["dead"] for r in "ABCD"], [0, 1, 2, 1])
        self.assertEqual([t[r]["ghosts"] for r in "ABCD"], [0, 0, 1, 0])
        self.assertEqual([t[r]["answer_changed"] for r in "ABCD"], [0, 1, 3, 1])

    def test_tombstones(self):
        ts = {t["id"]: t for t in self.g["tombstones"]}
        self.assertEqual(set(ts), {"q2", "q3"})
        self.assertEqual(ts["q2"]["died_at"], "B")
        self.assertEqual(ts["q2"]["right_again_at"], [])
        self.assertEqual(ts["q3"]["died_at"], "C")
        self.assertEqual(ts["q3"]["right_again_at"], ["D"])

    def test_ghosts(self):
        self.assertEqual(self.g["ghosts"], [{"id": "q4", "reference_said": 2, "right_at": ["C"]}])

    def test_missing_question_refuses(self):
        del self.by["C"]["q2"]
        with self.assertRaises(ValueError):
            bury_rows(self.by, ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
