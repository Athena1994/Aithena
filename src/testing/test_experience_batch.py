import copy
import unittest

import numpy as np
import torch
from aithena.qlearning.markov.experience import ExperienceBatch, Experience
from aithena.qlearning.markov.state import make_descriptor


class TestExperienceBatch(unittest.TestCase):
    def setUp(self):
        self.batch = ExperienceBatch(
            old_state={'a': torch.rand((2, 42, 69)),
                       'b': torch.rand((2, 42, 69))},
            action=np.array([3, 5]),
            reward=np.array([234, -12]),
            new_state={'a': torch.rand((2, 42, 69)),
                       'b': torch.rand((2, 42, 69))},
            terminal=np.array([False, True]),
        )
        self.experience1 = Experience(
            old_state={k: self.batch.old_states[k][0] for k in ('a', 'b')},
            action=self.batch.actions[0],
            reward=self.batch.rewards[0],
            new_state={k: self.batch.new_states[k][0] for k in ('a', 'b')},
        )
        self.experience2 = Experience(
            old_state={k: self.batch.old_states[k][1] for k in ('a', 'b')},
            action=self.batch.actions[1],
            reward=self.batch.rewards[1],
            new_state={k: self.batch.new_states[k][1] for k in ('a', 'b')},
            terminal=True,
        )
        self.experience3 = Experience(
            old_state={'a': torch.rand((42, 69)),
                       'b': torch.rand((42, 69))},
            action=42,
            reward=69,
            new_state={'a': torch.rand((42, 69)),
                       'b': torch.rand((42, 69))},
        )

    def test_create_empty(self):
        empty_batch = ExperienceBatch.create_empty(5, {'a': (42, 69),
                                                       'b': (42, 69)},
                                                   cuda=False)
        self.assertEqual(empty_batch.size, (5,))
        self.assertTrue(torch.all(empty_batch.actions == 0))
        self.assertTrue(torch.all(empty_batch.rewards == 0))
        for key in empty_batch.old_states:
            self.assertTrue(torch.all(empty_batch.old_states[key] == 0))
            self.assertTrue(torch.all(empty_batch.new_states[key] == 0))

        self.assertEqual(make_descriptor(empty_batch.old_states),
                         {'a': (5, 42, 69), 'b': (5, 42, 69)})

    def test_state_descriptor(self):
        self.assertEqual(self.batch.state_descriptor,
                         {'a': (42, 69), 'b': (42, 69)})

    def test_state_keys(self):
        self.assertEqual(self.batch.state_keys, {'a', 'b'})

    def test_experiences(self):
        experiences = self.batch.experiences
        self.assertEqual(len(experiences), 2)
        self.assertIn(self.experience1, experiences)
        self.assertIn(self.experience2, experiences)
        self.assertNotIn(self.experience3, experiences)

    def test_get_item(self):
        self.assertEqual(self.batch[0], self.experience1)
        self.assertEqual(self.batch[1], self.experience2)
        self.assertNotEqual(self.batch[1], self.experience3)

    def test_set_item(self):
        self.batch[1] = self.experience3
        self.assertEqual(self.batch[1], self.experience3)

    def test_batch_length(self):
        self.assertEqual(len(self.batch), 2)

    def test_get_experience_by_index(self):
        self.assertEqual(self.batch[0], self.experience1)
        self.assertEqual(self.batch[1], self.experience2)

    def test_iterate_over_batch(self):
        experiences = [exp for exp in self.batch]
        self.assertEqual(experiences, [self.experience1, self.experience2])

    def test_slice(self):
        batch = ExperienceBatch.create_empty(5, {'a': (42, 69), 'b': (42, 69)},
                                             cuda=False)
        batch[0] = self.experience1
        batch[1] = self.experience2
        batch[2] = self.experience3
        sliced_batch = batch[:2]
        self.assertEqual(len(sliced_batch), 2)
        self.assertEqual(sliced_batch[0], self.experience1)
        self.assertEqual(sliced_batch[1], self.experience2)

        sliced_batch = batch[1:4]
        self.assertEqual(len(sliced_batch), 3)
        self.assertEqual(sliced_batch[0], self.experience2)
        self.assertEqual(sliced_batch[1], self.experience3)
        self.assertEqual(sliced_batch[2], batch[3])

    def test_sample_experiences(self):
        batch = ExperienceBatch.create_empty(5, {'a': (42, 69), 'b': (42, 69)},
                                             cuda=False)
        batch[0] = self.experience1
        batch[1] = self.experience2
        batch[2] = self.experience3

        sampled_batch = batch.sample(np.array([0, 2]))
        self.assertEqual(len(sampled_batch), 2)
        self.assertEqual(sampled_batch[0], self.experience1)
        self.assertEqual(sampled_batch[1], self.experience3)

        sample_shape = (16, 64)

        ix = np.random.choice(
            range(3), sample_shape,
            replace=True)

        sampled_batch = batch.sample(ix)
        self.assertEqual(sampled_batch.size, sample_shape)

    def test_flatten(self):
        desc = {'a': (42, 69), 'b': (42, 69)}
        batch = ExperienceBatch.create_empty((5, 10, 10, 10),
                                             copy.deepcopy(desc), False)

        self.assertEqual(batch.size, (5, 10, 10, 10))
        self.assertEqual(batch.actions.shape, (5, 10, 10, 10))
        self.assertEqual(batch.old_states['a'].shape, (5, 10, 10, 10, 42, 69))
        self.assertDictEqual(batch.state_descriptor, desc)

        flat_batch = batch.flatten(3)
        self.assertEqual(flat_batch.size, (500, 10))
        self.assertEqual(flat_batch.actions.shape, (500, 10))
        self.assertEqual(flat_batch.old_states['a'].shape, (500, 10, 42, 69))
        self.assertEqual(batch.size, (5, 10, 10, 10))
        self.assertEqual(batch.actions.shape, (5, 10, 10, 10))
        self.assertEqual(batch.old_states['a'].shape, (5, 10, 10, 10, 42, 69))

        flat_batch = flat_batch.flatten()
        self.assertEqual(flat_batch.size, (5000, ))
        self.assertEqual(flat_batch.actions.shape, (5000,))
        self.assertEqual(flat_batch.old_states['a'].shape, (5000, 42, 69))

    def test_multi_dim_slicing(self):
        batch = ExperienceBatch.create_empty((5, 10),
                                             {'a': (42, 69), 'b': (42, 69)},
                                             cuda=False)
        self.assertEqual(batch.size, (5, 10))
        self.assertEqual(batch.old_states['a'].shape, (5, 10, 42, 69))

        sub_batch = batch[1]
        self.assertEqual(sub_batch.size, (10,))
        self.assertEqual(sub_batch.old_states['a'].shape, (10, 42, 69))
