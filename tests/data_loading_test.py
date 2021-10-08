import unittest
import epicas

class StructuredDataTest(unittest.TestCase):
    def test_static_data(self):
        population = epicas.StructuredData(
            'demo/datasets/population.csv',
            location = 'location',
            usecols = ['location', 'population']
            )

        self.assertEqual(
            population.variables,
            {'static': ['location', 'population'], 'time_series': None}
        )

    def test_time_series(self):
        jhu = epicas.StructuredData(
            'demo/datasets/covid.xz',
            location = 'FIPS',
            date = 'date',
            incidence = 'confirmed_cases',
            )

        self.assertEqual(
            jhu.variables,
            {'static': None,
            'time_series': ['location', 'date', 'incidence', 'confirmed_cases_norm']
            }
        )

    def test_merging(self):
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

        test = population + jhu + mobility

        self.assertEqual(
            len(test.variables['static']),
            2
        )

        self.assertEqual(
            len(test.variables['time_series']),
            6
        )

if __name__ == '__main__':
    unittest.main()
