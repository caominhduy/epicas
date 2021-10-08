import unittest
import epicas

class FeatureEngineeringTest(unittest.TestCase):
    def test_EpiData(self):
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

        merged = merged.lag_reduction(
            subset=['fb_movement_change', 'fb_stationary'], sliding_window=21,
            verbose=False
        )

        self.assertEqual(merged.df.size, 9216942)

if __name__ == '__main__':
    unittest.main()
