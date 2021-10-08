import unittest
import epicas

class SingleModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        population = epicas.StructuredData(
            'demo/datasets/population.csv',
            location = 'location',
            usecols = ['location', 'population']
            )

        jhu = epicas.StructuredData(
            'demo/datasets/covid.xz',
            location = 'FIPS',
            date = 'date',
            incidence = 'confirmed_cases',
            )

        mobility = epicas.StructuredData(
            'demo/datasets/mobility.csv.gz',
            location = 'FIPS',
            date = 'date'
        )

        merged = jhu + population + mobility

        merged = epicas.EpiData(merged, y='incidence', disease='covid19')

        merged = merged.imputation().target_to_ma(window=3)

        merged = merged.normalization(
            subset=['population', 'fb_movement_change', 'fb_stationary']
        )

        self.merged = merged.lag_reduction(
            subset=['fb_movement_change', 'fb_stationary'], sliding_window=21,
            verbose=False
        )

    def test_GRU(self):
        model = epicas.AutoGRU(self.merged).load('tests/pretrained/hu_64/GRU')

        output = model.predict('2021-09-01')

        self.assertEqual(len(output.index), 66648)

    def test_LSTM(self):
        model = epicas.AutoLSTM(self.merged).load('tests/pretrained/hu_64/LSTM')

        output = model.predict('2021-09-01')

        self.assertEqual(len(output.index), 66648)

    def test_BiLSTM(self):
        model = epicas.AutoBiLSTM(self.merged).load('tests/pretrained/hu_64/BiLSTM')

        output = model.predict('2021-09-01')

        self.assertEqual(len(output.index), 66648)

    def test_attention(self):
        model = epicas.AutoAttention(self.merged).load('tests/pretrained/hu_64/attention')

        output = model.predict('2021-09-01')

        self.assertEqual(len(output.index), 66648)

if __name__ == '__main__':
    unittest.main()
