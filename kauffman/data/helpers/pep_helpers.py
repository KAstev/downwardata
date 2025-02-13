import os
import sys
import requests
import numpy as np
import pandas as pd
import kauffman.constants as c
from kauffman.tools.etl import county_msa_cross_walk as cw

# https://www.census.gov/programs-surveys/popest.html

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 35000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def _make_header(df):
    df.columns = df.iloc[0]
    return df.iloc[1:]


def _row_format(row):
    return [row[0]] + [' '.join(row[1: -5])] + row[-5:]


def _obs_filter(df):
    return df.\
        query('fips != "FIPS"').\
        query('fips != "Code"').\
        reset_index(drop=True)


def _county_1980_1989():
    lst_1980, lst_1985 = [], []
    lst_1980_bool = 1

    lines_iter = iter(requests.get('https://www2.census.gov/programs-surveys/popest/tables/1980-1990/counties/totals/e8089co.txt').text.split('\n')[25:])
    while True:
        row = lines_iter.__next__().split()

        if not row:
            continue
        if row and row[0] == 'FIPS':
            lst_1980_bool = not lst_1980_bool

        if lst_1980_bool:
            if len(row) < 7:
                row = row + lines_iter.__next__().split()
            lst_1980.append(_row_format(row))
        else:
            if len(row) < 7:
                row = row + lines_iter.__next__().split()
            lst_1985.append(_row_format(row))
            if row[0] == '56045':
                break

    df_1980 = pd.DataFrame(lst_1980, columns=['fips', 'region', 'time1980', 'time1981', 'time1982', 'time1983', 'time1984']).\
        pipe(_obs_filter)
    df_1985 = pd.DataFrame(lst_1985, columns=['fips', 'region', 'time1985', 'time1986', 'time1987', 'time1988', 'time1989']).\
        pipe(_obs_filter).\
        drop('region', 1)

    return df_1980.\
        merge(df_1985, how='left', on='fips').\
        pipe(pd.wide_to_long, 'time', i='fips', j='year').\
        query(f'region not in {list(c.state_name_abb_dic.keys())}'). \
        assign(region=lambda x: x['region'].replace(r'Co\.', 'County', regex=True)).\
        reset_index(drop=False).\
        rename(columns={'time': 'population', 'year': 'time'}).\
        astype({'time': 'int', 'population': 'int'})


def _county_1990_1999():
    lst_1990 = []

    lines = requests.get('https://www2.census.gov/programs-surveys/popest/tables/1990-2000/counties/totals/99c8_00.txt').text.split('\n')[12:]
    for line in lines:
        row = line.split()

        if not row:
            break
        if row[1] == '49041':
            lst_1990.append(row[1:7] + [np.nan] + row[10:14] + [' '.join(row[15:])])
        elif row[1] == '50027':
            lst_1990.append(row[1:7] + [np.nan] + row[9:13] + [' '.join(row[14:])])
        else:
            lst_1990.append(row[1:12] + [' '.join(row[13:])])

    return pd.DataFrame(lst_1990, columns=['fips'] + ['time' + str(year) for year in range(1999, 1989, -1)] + ['region']).\
        pipe(pd.wide_to_long, 'time', i='fips', j='year').\
        query(f'region not in {list(c.state_name_abb_dic.keys())}').\
        reset_index().\
        assign(
            population=lambda x: x['time'].replace(',', '', regex=True)
        ).\
        drop('time', 1).\
        rename(columns={'year': 'time'}).\
        astype({'time': 'int', 'population': 'float'})


def _county_2000_2009():
    return pd.concat(
            [
                pd.DataFrame(requests.get('https://api.census.gov/data/2000/pep/int_population?get=GEONAME,POP,DATE_DESC&for=county:*&DATE_={0}'.format(date)).json()). \
                    pipe(_make_header).\
                    query('state != "72"').\
                    assign(
                        time=lambda x: 1998 + x['DATE_'].astype(int),
                        fips=lambda x: x['state'] + x['county'],
                        region=lambda x: x['GEONAME'].str.split(',').str[0]
                    ). \
                    rename(columns={'POP': 'population'}) \
                    [['fips', 'region', 'time', 'population']]
                for date in range(2, 12)
            ],
            axis=0
        ).\
        sort_values(['fips', 'time']).\
        reset_index(drop=True)


def _county_2010_2019():
    return pd.DataFrame(requests.get('https://api.census.gov/data/2019/pep/population?get=NAME,POP,DATE_CODE&for=county:*').json()). \
        pipe(_make_header). \
        query('state != "72"'). \
        astype({'DATE_CODE': 'int'}).\
        query('3 <= DATE_CODE').\
        assign(
            time=lambda x: 2007 + x['DATE_CODE'],
            fips=lambda x: x['state'] + x['county'],
            region=lambda x: x['NAME'].str.split(',').str[0]
        ). \
        rename(columns={'POP': 'population'}) \
        [['fips', 'region', 'time', 'population']].\
        sort_values(['fips', 'time']).\
        reset_index(drop=True)


## 1900 - 2000 code
def _format(df, astype_arg=None, query_arg=None, format_pop=True):
    df = df.astype(astype_arg) if astype_arg else df
    df = df.query(query_arg) if query_arg else df

    return df.reset_index(drop=False).\
        assign(
            POP=lambda x: x['POP'].replace(',', '', regex=True).astype(int) * 1000 if format_pop else x['POP'],
            region=lambda x: x['region'].map(c.abb_name_dic),
            fips=lambda x: x['region'].map(c.all_name_fips_dic)
        ) \
        [['fips', 'region', 'time', 'POP']]


def _get_data(year, url_code, lrange, mid=6, end=10, cols1=['region'], cols2=['region']):
    base_url = 'https://www2.census.gov/programs-surveys/popest/tables/1980-1990/state/asrh'
    lines = requests.get(f'{base_url}/{url_code}.txt').text.split('\n')

    return pd.DataFrame(
        [line.split() for line in lines[lrange[0][0]: lrange[0][1]]], 
        columns=cols1 + list(map(lambda x: f'POP{x}', range(year, year + mid)))
        ).\
        pipe(pd.wide_to_long, 'POP', i='region', j='time').\
        append(
            pd.DataFrame(
                [line.split() for line in lines[lrange[1][0]:lrange[1][1]]], 
                columns=cols2 + list(map(lambda x: f'POP{x}', range(year + mid, year + end)))
            ).\
            pipe(pd.wide_to_long, 'POP', i='region', j='time')
        )

def _state_1900_1989(year):
    url_codes = {1900: 'st0009ts', 1910: 'st1019ts_v2', 1920: 'st2029ts', 1930: 'st3039ts',
        1940: 'st4049ts', 1950: 'st5060ts', 1960: 'st6070ts', 1970: 'st7080ts', 1980: 'st8090ts'} 

    line_ranges = {
        **{year : [(23,72), (82,-1)] for year in [1900, 1910, 1920, 1930]},
        **{
            1940: [(21, 70), (79,-1)],
            1950: [(27, 78), (92, -3)],
            1960: [(24, 75), (86,-1)], 
            1970: [(14, 65), (67, -8)],
            1980: [(11, 62), (70, -1)]
        }
    }
    
    if year in [1900, 1910, 1920, 1930, 1940]:
        df = _get_data(year, url_codes[year], line_ranges[year]).\
            pipe(_format)
    elif year in [1950, 1960]:
        df = _get_data(year, url_codes[year], line_ranges[year], 5, 11, ['region', 'census']).\
            pipe(_format, query_arg=f'time < {year + 10}')
    elif year == 1970:
        df = _get_data(year, url_codes[year], line_ranges[year], 6, 11, ['id', 'region'], ['id', 'region']).\
            pipe(_format, astype_arg = {'POP': 'int'}, query_arg=f'time < {year + 10}', format_pop=False)
    elif year == 1980:
        df = _get_data(year, url_codes[year], line_ranges[year], 5, 11).\
            pipe(_format, astype_arg = {'POP': 'int'}, query_arg=f'time < {year + 10}', format_pop=False)
    
    return df

def _state_1990_1999():
    lines = requests.get('https://www2.census.gov/programs-surveys/popest/tables/1990-2000/state/totals/st-99-07.txt').text.split('\n')

    rows = []
    for row in lines[28: 79]:
        rows.append(
            row.split()[:2] + \
            [' '.join([element for element in row.split() if element not in row.split()[:2] + row.split()[-11:]])] + \
            row.split()[-11:]
        )
    return pd.DataFrame(rows, columns=['block', 'fips', 'region'] + list(map(lambda x: f'POP{x}', range(1999, 1989, -1))) + ['census']).\
        drop(['block', 'census'], 1). \
        pipe(pd.wide_to_long, 'POP', i='region', j='time'). \
        reset_index(drop=False) \
        [['fips', 'region', 'time', 'POP']]


def _state_2000_2009():
    url = 'https://api.census.gov/data/2000/pep/int_population?get=GEONAME,POP,DATE_&for=state:*'

    return pd.DataFrame(requests.get(url).json()). \
        pipe(_make_header). \
        rename(columns={'GEONAME': 'region', 'DATE_': 'date'}). \
        astype({'date': 'int'}). \
        query('2 <= date <= 11').\
        query('region not in ["Puerto Rico"]').\
        assign(
            time=lambda x: '200' + (x['date'] - 2).astype(str),
            fips=lambda x: x['region'].map(c.all_name_fips_dic),
        ) \
        [['fips', 'region', 'time', 'POP']]


def _state_2010_2019():
    url = 'https://api.census.gov/data/2019/pep/population?get=NAME,POP,DATE_CODE&for=state:*'

    return pd.DataFrame(requests.get(url).json()). \
        pipe(_make_header). \
        rename(columns={'NAME': 'region', 'DATE_CODE': 'date'}). \
        astype({'date': 'int'}). \
        query('3 <= date <= 12').\
        query('region not in ["Puerto Rico"]').\
        assign(
            time=lambda x: '20' + (x['date'] + 7).astype(str),
            fips=lambda x: x['region'].map(c.all_name_fips_dic),
        ) \
        [['fips', 'region', 'time', 'POP']]


def _us_1900_1999():
    return pd.read_csv(
            'https://www2.census.gov/programs-surveys/popest/tables/1900-1980/national/totals/popclockest.txt',
            delim_whitespace=True,
            skiprows=9,
            skipfooter=25,
            usecols=[2, 3],
            names=['time', 'POP'],
            converters={'POP': lambda x: x.replace(',', '')},
        ). \
        assign(
            region='United States',
            fips='00'
        )


def _us_2000_2009():
    url = 'https://api.census.gov/data/2000/pep/int_population?get=GEONAME,POP,DATE_&for=us:1'

    return pd.DataFrame(requests.get(url).json()). \
        pipe(_make_header). \
        rename(columns={'GEONAME': 'region', 'DATE_': 'date'}). \
        astype({'date': 'int'}). \
        query('2 <= date <= 11').\
        query('region not in ["Puerto Rico"]').\
        assign(
            time=lambda x: '200' + (x['date'] - 2).astype(str),
            fips=lambda x: x['region'].map(c.all_name_fips_dic),
        ) \
        [['fips', 'region', 'time', 'POP']]


def _us_2010_2019():
    url = 'https://api.census.gov/data/2019/pep/population?get=NAME,POP,DATE_CODE&for=us:*'

    return pd.DataFrame(requests.get(url).json()). \
        pipe(_make_header). \
        rename(columns={'NAME': 'region', 'DATE_CODE': 'date'}). \
        astype({'date': 'int'}). \
        query('3 <= date <= 12').\
        query('region not in ["Puerto Rico"]').\
        assign(
            time=lambda x: '20' + (x['date'] + 7).astype(str),
            fips=lambda x: x['region'].map(c.all_name_fips_dic),
        ) \
        [['fips', 'region', 'time', 'POP']]


def _pep_data_create(region):
    if region == 'county':
        df = pd.concat(
            [
                f() for f in [_county_1980_1989, _county_1990_1999, _county_2000_2009, _county_2010_2019]
            ],
            axis=0
        ).\
            assign(region=lambda x: x['fips'].map(c.all_fips_name_dic))  # todo: can clean region up in the above functions, and also the functions at the return statement below: put those in _county fucntions or below?
    elif region == 'msa':
        df = _pep_data_create('county').\
            pipe(cw, 'fips'). \
            assign(fips=lambda x: x['fips_msa'].astype(int).astype(str)) \
            [['fips', 'region', 'time', 'population']]
    elif region == 'state':
        df = pd.concat(
                [_state_1900_1989(year) for year in range(1900,1981,10)] + 
                [f() for f in [_state_1990_1999, _state_2000_2009, _state_2010_2019]],
                axis=0
            )
    else:
        df = pd.concat(
            [
                f() for f in [_us_1900_1999, _us_2000_2009, _us_2010_2019]
            ],
            axis=0
        )

    return df. \
        sort_values(['fips', 'time']). \
        reset_index(drop=True).\
        rename(columns={'POP': 'population'}).\
        astype({'population': 'float', 'time': 'int'}) \
        [['fips', 'region', 'time', 'population']]
