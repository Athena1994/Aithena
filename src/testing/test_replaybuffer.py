import unittest

from aithena.qlearning.markov.state import create_random_state
from aithena.qlearning.markov.replay_buffer \
    import Experience, ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.state_desc = {'a': (2,),
                           'b': (4, 2),
                           'c': (1, 3, 2)}
        self.replay_buffer = ReplayBuffer(128, self.state_desc)

        self.experience_factory = lambda i: Experience(
            old_state=create_random_state(self.state_desc),
            action=i,
            reward=float(i),
            new_state=create_random_state(self.state_desc)
        )

    def test_initialization(self):
        self.assertEqual(self.replay_buffer.size, 0)
        self.assertEqual(len(self.replay_buffer), 0)
        self.assertTrue(self.replay_buffer.is_empty)
        self.assertFalse(self.replay_buffer.is_full)
        self.assertEqual(self.replay_buffer.capacity, 128)
        self.assertEqual(self.replay_buffer.state_descriptor, self.state_desc)
        self.assertTrue(len(self.replay_buffer.buffer) == 0)
        self.assertTrue(len(self.replay_buffer.experiences) == 0)

    def test_add_experience(self):
        for i in range(128):
            exp = self.experience_factory(i)
            self.replay_buffer.add_experience(exp, 0.42)
            self.assertEqual(self.replay_buffer.size, i + 1)
            self.assertTrue(len(self.replay_buffer.buffer) == i+1)
            self.assertTrue(len(self.replay_buffer.experiences) == i+1)
            self.assertEqual(self.replay_buffer.buffer[i], exp)

        self.assertEqual(self.replay_buffer.size, 128)
        self.assertEqual(len(self.replay_buffer), 128)
        self.assertFalse(self.replay_buffer.is_empty)
        self.assertTrue(self.replay_buffer.is_full)

        # Check if the experiences are stored correctly
        for i in range(128):
            exp = self.replay_buffer.buffer[i]
            self.assertEqual(exp.action, i)
            self.assertEqual(exp.reward, float(i))

        for i in range(128):
            exp = self.experience_factory(i)
            self.replay_buffer.add_experience(exp, 1)
            self.assertEqual(self.replay_buffer.size, 128)
            self.assertTrue(len(self.replay_buffer.buffer) == 128)
            self.assertTrue(len(self.replay_buffer.experiences) == 128)
            self.assertIn(exp, self.replay_buffer.buffer)

    def test_weights(self):
        for i in range(128):
            self.replay_buffer.add_experience(self.experience_factory(i), .42)
            self.assertEqual(self.replay_buffer._weights[i], .42)
            self.assertAlmostEqual(self.replay_buffer._weight_sum, (i + 1)*.42)

        self.replay_buffer.add_experience(self.experience_factory(i), 1)
        self.assertIn(1, self.replay_buffer._weights)
        self.assertAlmostEqual(self.replay_buffer._weight_sum, 127 * .42+1)

    def test_clear(self):
        for i in range(128):
            self.replay_buffer.add_experience(self.experience_factory(i), 0.42)

        self.assertEqual(self.replay_buffer.size, 128)
        self.assertFalse(self.replay_buffer.is_empty)
        self.assertTrue(self.replay_buffer.is_full)

        self.replay_buffer.clear()

        self.assertEqual(self.replay_buffer.size, 0)
        self.assertTrue(self.replay_buffer.is_empty)
        self.assertFalse(self.replay_buffer.is_full)
        self.assertEqual(len(self.replay_buffer.buffer), 0)
        self.assertEqual(len(self.replay_buffer.experiences), 0)
        self.assertEqual(self.replay_buffer._weight_sum, 0)

    def test_sample_experiences(self):
        for i in range(128):
            self.replay_buffer.add_experience(self.experience_factory(i), 0.42)

        res = self.replay_buffer.sample_experiences((16, 64), True)
        self.assertEqual((16, 64), res.size)
