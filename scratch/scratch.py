import joblib
import pandas as pd
import kauffman.constants as c
from kauffman import raw_kese_formatter

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def scratch_panel_to_alley():
    df = pd.read_csv(c.filenamer('../scratch/jobs_indicators_sample_all.csv')).\
        rename(columns={'Age': 'Age of Business'})
    df.loc[df['Age of Business'] == '0-1', 'Age of Business'] = 'Ages 0 to 1'
    df.loc[df['Age of Business'] == '2-3', 'Age of Business'] = 'Ages 2 to 3'
    df.loc[df['Age of Business'] == '4-5', 'Age of Business'] = 'Ages 4 to 5'
    df.loc[df['Age of Business'] == '6-10', 'Age of Business'] = 'Ages 6 to 10'
    df.loc[df['Age of Business'] == '11+', 'Age of Business'] = 'Ages 11+'

    df_overall = df.loc[df['Age of Business'] == 'Ages 0 to 1']
    df_overall.loc[:, 'Age of Business'] = 'overall'

    df_in = df.\
        append(df_overall).\
        sort_values(['fips', 'year', 'Age of Business'])

    df_out = df_in.pub.panel_to_alley(['Age of Business'], 'Age Share of Employment')
    print(df_out.head(65))


# todo: create a repo that does this or something...I need to remember what to do once a year when we get this request
def scratch_kese_to_panel():
    df = raw_kese_formatter(c.filenamer('../scratch/Kauffman Indicators Data State 1996_2019_v3.xlsx'), c.filenamer('../scratch/Kauffman Indicators Data National 1996_2020.xlsx'))
    df.to_csv('/Users/thowe/Downloads/kese_2020_download.csv', index=False)
    print(df.head())

    for indicator in ['rne', 'ose', 'sjc', 'ssr', 'zindex']:
        df.pub.download_to_alley_formatter(['type', 'category'], indicator).\
            to_csv(f'/Users/thowe/Downloads/kese_2020_{indicator}.csv', index=False)


def plot_maps():
    # pd.read_csv('/Users/thowe/Downloads/ba_state_perc_change.csv'). \
    #     assign(fips=lambda x: x['Region'].map(c.state_dic_temp)). \
    #     pub.choro_map('Percentage Change')

    # import kauffman.bfs as bfs
    # bfs.get_data(['BA_BA'], obs_level='state'). \
    #     query('time == 2019'). \
    #     assign(fips=lambda x: x['region'].map(c.state_dic_temp)). \
    #     pub.choro_map('BA_BA', 'Business Applications 2019', 'Business Applications')

    import kauffman.data.helpers.pep_helpers as pep
    pep.get_data('county'). \
        query('time == "2019"'). \
        astype({'population': 'int'}). \
        pub.choro_map('population', 'County Population 2019', 'Population', write='/Users/thowe/Downloads/scratch.png', range_factor=.02)
    #     pipe(joblib.dump, '/Users/thowe/Downloads/scratch.pkl')
    joblib.load('/Users/thowe/Downloads/scratch.pkl').\
        query('time == "2019"'). \
        astype({'population': 'int'}). \
        pub.choro_map('population', 'County Population 2019', 'Population', write='/Users/thowe/Downloads/scratch.png', range_factor=.02)


def msa_plot():
    # qwi.get_data('msa', ['Emp'], start_year=2016, end_year=2016, annualize=True).to_csv('/Users/thowe/Downloads/scratch.csv', index=False)
    df = pd.read_csv('/Users/thowe/Downloads/scratch.csv').\
        query('firmage == 1').\
        astype({'fips': 'str'}).\
        pub.choro_map('msa', 'Emp', 'MSA Startup Employment 2016', 'Emp', write=False, range_factor=.5)

def main():
    # scratch_panel_to_alley()
    scratch_kese_to_panel()
    # plot_maps()
    # msa_plot()

if __name__ == '__main__':
    main()