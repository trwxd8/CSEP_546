import unittest
import DecisionTreeModel as dtm


class TestDecisionTreeModel(unittest.TestCase):

    def test_entropy_S(self):

        # Mitchell, page 56, when all values are 0
        yTrains = [0] * 10
        s = dtm.get_entropy_S(yTrains)
        self.assertTrue(s == 0)

        # Mitchell, page 56, when all values are 1
        yTrains = [1] * 10
        s = dtm.get_entropy_S(yTrains)
        self.assertTrue(s == 0)

        # Mitchell, page 56, when num(0s) == num(1s)
        yTrains = [1] * 5 + [0] * 5
        s = dtm.get_entropy_S(yTrains)
        self.assertTrue(s == 1)

        # case - Mitchell, Chapter 3, page 56
        yTrains = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        s = dtm.get_entropy_S(yTrains)
        self.assertAlmostEqual(s, 0.940, 3)

    def test_get_entropy_for_feature(self):

        # Mitchell, page 58
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        feature_dict = dtm.get_feature_dict(wind, play_tennis)
        es = dtm.get_entropy_for_feature(feature_dict)
        for entropy, expected in zip(es, [0.811, 1.0]):
            self.assertAlmostEqual(entropy, expected, 3)

        # Mitchell, page 57 (one of p is 1.0, entropy should be 0)
        feature_dict = {0: {0: 10, 1: 0}, 1: {0: 0, 1: 10}}
        es = dtm.get_entropy_for_feature(feature_dict)
        for entropy, expected in zip(es, [0.0, 0.0]):
            self.assertAlmostEqual(entropy, expected, 3)

    def test_get_information_gain(self):

        # Mitchell, page 58
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        gain = dtm.get_information_gain(wind, play_tennis)
        self.assertAlmostEqual(gain, 0.048, 3)

    def test_get_information_gains(self):
        """
        Construct xTrains data to have same data structure
        from original framework.
        """
        # Mitchell, page 59 and 60
        humidity = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]  # high = 0
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        xTrains = [[h, w] for h, w in zip(humidity, wind)]
        gains = dtm.get_information_gains(xTrains, play_tennis)
        for gain, expected in zip(gains, [0.151, 0.048]):
            self.assertAlmostEqual(gain, expected, 2)

    def test_get_split(self):

        humidity = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]  # high = 0
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

        # pick humidity as a feature to split
        xTrains = [[h, w] for h, w in zip(humidity, wind)]
        node = dtm.get_split(xTrains, play_tennis)
        self.assertTrue(node['index'] == 0)

        # swap the orders in feature.
        xTrains = [[w, h] for w, h in zip(wind, humidity)]
        node = dtm.get_split(xTrains, play_tennis)
        self.assertTrue(node['index'] == 1)

    def test_split(self):

        # Given a node and grow tree recursively until it meets the stop requirements.
        humidity =    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]  # high = 0
        wind =        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        xTrains = [[h, w] for h, w in zip(humidity, wind)]
        node = dtm.get_split(xTrains, play_tennis)
        dtm.split(node, min_to_stop=1)
        dtm.print_tree(node)

    def test_predict(self):

        # Given a node and grow tree recursively until it meets the stop requirements.
        humidity = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]  # high = 0
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        xTrains = [[h, w] for h, w in zip(humidity, wind)]
        model = dtm.DecisionTreeModel()
        model.fit(xTrains, play_tennis, min_to_stop=1)
        expected_predictions = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
        # give me perfect prediction
        predictions = model.predict(xTrains)
        self.assertTrue(predictions == expected_predictions)


if __name__ == '__main__':
    unittest.main()