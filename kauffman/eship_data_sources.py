
import pandas as pd
import kauffman.constants as c
from kauffman.bfs_helpers import _bfs_data_create
from kauffman.bds_helpers import _bds_data_create

def bfs(series_lst, obs_level='all', seasonally_adj=True, annualize=False):
    """
    series_lst: lst

        Variables:
            BA_BA: 'Business Applications'
            BA_CBA: 'Business Applications from Corporations'
            BA_HBA: 'High-Propensity Business Applications'
            BA_WBA: 'Business Applications with Planned Wages'
            BF_BF4Q: 'Business Formations within Four Quarters'
            BF_BF8Q: 'Business Formations within Eight Quarters'
            BF_PBF4Q: Projected Business Formations within Four Quarters
            BF_PBF8Q: Projected Business Formations within Eight Quarters
            BF_SBF4Q: Spliced Business Formations within Four Quarter
            BF_SBF8Q: Spliced Business Formations within Eight Quarters
            BF_DUR4Q: Average Duration (in Quarters) from Business Application to Formation within Four Quarters
            BF_DUR8Q: Average Duration (in Quarters) from Business Application to Formation within Eight Quarters


    """

    if type(obs_level) == list:
        region_lst = obs_level
    else:
        if obs_level == 'us':
            region_lst = ['US']
        elif obs_level == 'state':
            region_lst = c.states
        else:
            region_lst = ['US'] + c.states

    return pd.concat(
            [
                _bfs_data_create(region, series_lst, seasonally_adj, annualize)
                for region in region_lst
            ],
            axis=0
        ).\
        reset_index(drop=True) \
        [['fips', 'region', 'time'] + series_lst]


def bds(series_lst, obs_level='all'):
    """
    series_lst: lst; see https://www.census.gov/content/dam/Census/programs-surveys/business-dynamics-statistics/BDS_Codebook.pdf or https://api.census.gov/data/timeseries/bds/variables.html
        FIRM: Number of firms
        FAGE: Firm age code
        NET_JOB_CREATION: Number of net jobs created from expanding/contracting and opening/closing establishments during the last 12 months

    obs_level: str or lst
        all:
        us:
        state:
        county:
        list of regions according to fips code

    first year available is 1978, last year is 2018
    """

    if type(obs_level) == list:
        region_lst = obs_level
    else:
        if obs_level in ['us', 'state', 'county']:
            region_lst = [obs_level]
        else:
            region_lst = ['us', 'state', 'county']

    return pd.concat(
            [
                _bds_data_create(series_lst, region)
                for region in region_lst
            ],
            axis=0
        ). \
        reset_index(drop=True) \
        [['fips', 'region', 'time'] + series_lst]
