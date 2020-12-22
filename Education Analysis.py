import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from functools import reduce


def main():
    #                           READ IN DATA
    # read in data from .csv format
    os.chdir('~/Data')
    raw_edu_data = pd.read_csv('education_data.csv')
    wage_data = pd.read_csv('wage_data.csv')
    raw_col_data = pd.read_csv('COLI.csv')
    inf_data = pd.read_csv('inf_data.csv').rename(columns={'Year': 'YEAR'})

    #                           CLEAN DATA
    # replace misspellings in education data
    raw_edu_data.replace(['\xa0 United States', '  United States'], 'United States', inplace=True)

    # re-format cost-of-living data
    raw_col_data.dropna(inplace=True)
    col_data = pd.DataFrame(columns=['YEAR', 'COLI', 'STATE'])
    for year in range(2008, 2019):
        new_data = raw_col_data[[str(year), 'GeoName']].assign(YEAR=year)
        new_data.rename(columns={str(year): 'COLI', 'GeoName': 'STATE'}, inplace=True)
        col_data = pd.concat([col_data, new_data], sort=True)

    #                           IMPUTE EDU DATA
    # add in empty row for each missing observation
    years = [1997, 2000, 2001]
    for year in years:
        empty_data = pd.DataFrame(columns=raw_edu_data.columns)
        empty_data = empty_data.assign(STATE=raw_edu_data['STATE'].drop_duplicates(), YEAR=year)
        raw_edu_data = pd.concat([raw_edu_data, empty_data]).reset_index(drop=True)

    # create KNN imputer object
    imp = KNNImputer(n_neighbors=5, weights="distance")

    # impute missing data by state
    by_state = raw_edu_data.groupby('STATE')
    for state, frame in by_state:
        filled_data = pd.DataFrame(columns=frame.columns.drop('STATE'),
                                   data=imp.fit_transform(frame.drop(columns=['STATE'])))
        filled_data.loc[:, 'STATE'] = state
        if state == list(by_state.groups)[0]:
            edu_data = filled_data
        else:
            edu_data = pd.concat([edu_data, filled_data])

    #                           PLOT IMPUTATIONS
    nc = edu_data.query('STATE == "North Carolina"')
    plt.scatter(nc[~nc['YEAR'].isin([1997, 2000, 2001])]['YEAR'], nc[~nc['YEAR'].isin([1997, 2000, 2001])]['TOTAL'], color='g');
    plt.scatter(nc[nc['YEAR'].isin([1997, 2000, 2001])]['YEAR'], nc[nc['YEAR'].isin([1997, 2000, 2001])]['TOTAL'], color='b');
    plt.title('Actual and Imputed Total Spending per Pupil, North Carolina');
    plt.legend(['Actual', 'Imputed'])

    #                           DEFLATE
    # deflate each column representing dollars in education data
    edu_columns = ['TOTAL', 'SW', 'EB', 'INST_TOT', 'INST_SW', 'INST_EB']
    edu_data = pd.merge(edu_data, inf_data, on='YEAR')
    for col in edu_columns:
        edu_data.loc[:, col] = edu_data[col] * 100 / edu_data['Annual']
    edu_data.drop(columns='Annual', inplace=True)

    # deflate median annual wage in wage data
    wage_data = pd.merge(wage_data, inf_data, on='YEAR')
    wage_data.loc[:, 'A_MEDIAN'] = wage_data['A_MEDIAN'] * 100 / wage_data['Annual']
    wage_data.drop(columns='Annual', inplace=True)

    #                           REMOVE DC
    for data in edu_data, wage_data, col_data:
        data.query('STATE != "District of Columbia"', inplace=True)

    #                           MERGE
    raw_data = [edu_data, wage_data, col_data]
    raw_data = reduce(lambda left, right: pd.merge(left, right, on=['STATE', 'YEAR']), raw_data)

    #                           PLOT PARALLEL COORDINATES
    # filter for just statewide median wages, calculate means by state
    plot_data = raw_data.query('OCC_CAT == "00"').groupby('STATE').mean()

    # create plot
    fig = go.Figure(data=go.Parcoords(
                        line=dict(color=plot_data['COLI'],
                                  showscale=True),
                        dimensions=list([
                                   dict(range=[2500, 8500],
                                        label="Total Spending per Pupil",
                                        values=plot_data['TOTAL']),
                                   dict(range=[1500, 6000],
                                        label="Total Spending per Pupil, Instruction",
                                        values=plot_data['INST_TOT']),
                                   dict(range=[12000, 20000],
                                        label="Median Wage",
                                        values=plot_data['A_MEDIAN'])
    ])))

    fig.show()

    #                           MERGE FOR ANALYSIS
    # create comparison year column
    wage_data.rename(columns={'YEAR': 'COMP_YEAR'}, inplace=True)
    col_data.rename(columns={'YEAR': 'COMP_YEAR'}, inplace=True)
    edu_data.loc[:, 'COMP_YEAR'] = edu_data['YEAR'] + 10

    # convert independent, dependent variables to logs
    edu_data.loc[:, edu_columns] = np.log(edu_data.loc[:, edu_columns])
    wage_data.loc[:, 'A_MEDIAN'] = np.log(wage_data['A_MEDIAN'])

    # merge all data
    data = [edu_data, wage_data, col_data]
    data = reduce(lambda left, right: pd.merge(left, right, on=['STATE', 'COMP_YEAR']), data)

    #                           RUN GENERAL REGRESSION
    data.loc[:, 'CONST'] = 1
    test_data = data.query('OCC_CAT == "00"')
    X = test_data[['TOTAL', 'COLI', 'CONST']]
    y = test_data['A_MEDIAN']
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    #                           PLOT GENERAL REGRESSION
    fig, ax = plt.subplots()
    fig = sm.graphics.plot_fit(results, 0, ax=ax)
    ax.set_ylabel('ln(Median Wage)')
    ax.set_xlabel('ln(Per Pupil Spending 10 Years Prior)')
    ax.set_title('Raw and Fitted Values: Median Wage vs Per Pupil Spending')
    ax.legend(['Raw', 'Fitted'])

    #                           RUN REGRESSION BY DELAY
    # create output table to store results
    output_table = pd.DataFrame(columns=['YEAR_DELAY', 'COEFFICIENT', 'T-STAT', 'R^2'])
    year_delays = range(6, 16, 2)

    # loop through each delay to run regression
    for delay in year_delays:
        edu_data.loc[:, 'COMP_YEAR'] = edu_data['YEAR'] + delay
        data = [edu_data, wage_data, col_data]
        data = reduce(lambda left, right: pd.merge(left, right, on=['STATE', 'COMP_YEAR']), data)
        data.loc[:, 'CONST'] = 1
        test_data = data.query('OCC_CAT == "00"')
        X = test_data[['TOTAL', 'COLI', 'CONST']]
        y = test_data['A_MEDIAN']
        model = sm.OLS(y, X)
        results = model.fit()
        output_table = output_table.append(pd.DataFrame(columns=output_table.columns,
                                                        data=[[delay,
                                                               results.params[0],
                                                               results.tvalues[0],
                                                               results.rsquared]]))
    print(output_table.set_index('YEAR_DELAY'))

    #                           FILTER INDUSTRY BY WAGE
    # merge data with 10-year delay between education spending and wages/COLI
    edu_data.loc[:, 'COMP_YEAR'] = edu_data['YEAR'] + 10
    data = [edu_data, wage_data, col_data]
    data = reduce(lambda left, right: pd.merge(left, right, on=['STATE', 'COMP_YEAR']), data)

    # filter industry categories by average of median wages in each state in each year.
    top_5 = data.groupby('OCC_CAT').mean().sort_values(by='A_MEDIAN', ascending=False).index.to_list()[:5]
    bottom_5 = data.groupby('OCC_CAT').mean().sort_values(by='A_MEDIAN', ascending=False).index.to_list()[-5:]

    print(f"Top five occupational categories by wage: {top_5}")
    print(f"Bottom five occupational categories by wage: {bottom_5}")

    #                           RUN REGRESSION BY INDUSTRY
    # add constant for regression
    data.loc[:, 'CONST'] = 1

    # create output table to fill with results
    output_table = pd.DataFrame(columns=['TIER', 'INDUSTRY', 'COEFFICIENT', 'T-STAT', 'R^2'])

    # loop through each group
    for tier in ['Top 5', 'Bottom 5']:
        if tier == 'Top 5':
            occ_list = top_5
        else:
            occ_list = bottom_5
        # loop through each occupational category
        for occupation in occ_list:
            test_data = data.query('OCC_CAT == @occupation')
            X = test_data[['TOTAL', 'COLI', 'CONST']]
            y = test_data['A_MEDIAN']
            model = sm.OLS(y, X)
            results = model.fit()
            output_table = output_table.append(pd.DataFrame(columns=output_table.columns,
                                                            data=[[tier,
                                                                   occupation,
                                                                   results.params[0],
                                                                   results.tvalues[0],
                                                                   results.rsquared]]))
    print(output_table.set_index(['TIER', 'INDUSTRY']))

    #                           PLOT RESULTS
    # create list of professions, sorted by median income
    sorted_professions = data.groupby('OCC_CAT').mean().sort_values(by='A_MEDIAN', ascending=False).index.to_list()

    # create empty table to hold results for plotting
    plot_table = pd.DataFrame(columns=['INDUSTRY', 'IND_WAGE', 'COEFFICIENT', 'P-VALUE'])

    # run regression for each occupation
    for occupation in sorted_professions:
        test_data = data.query('OCC_CAT == @occupation')
        ind_wage = np.exp(test_data['A_MEDIAN'].mean())
        X = test_data[['TOTAL', 'COLI', 'CONST']]
        y = test_data['A_MEDIAN']
        model = sm.OLS(y, X)
        results = model.fit()
        # add results of regression to plot table
        plot_table = plot_table.append(pd.DataFrame(columns=plot_table.columns,
                                                                data=[[occupation,
                                                                       ind_wage,
                                                                       results.params[0],
                                                                       results.pvalues[0]]]))

    # create plot
    plt.scatter(plot_table['IND_WAGE'], plot_table['COEFFICIENT'], s=pd.cut(plot_table['P-VALUE'], [0, 0.01, 0.05, 1], labels=[25, 15, 5]))
    plt.xlabel('Median Wage of Employment Category')
    plt.ylabel('Coefficient of Education Spending on Earnings')
    plt.title('Education Elasticity of Earnings by Median Occupational Wage')


if __name__ == '__main__':
    main()
