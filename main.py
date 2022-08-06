import helpers
import notebooks
import prepareData


def main():
    dictionaries_array = [{"stationCode": 250190390, "cityName": 'MpKrak'},
                          {"stationCode": 252200150, "cityName": 'MzWar'},
                          {"stationCode": 252210160, "cityName": 'MzWar'},
                          {"stationCode": 349220695, "cityName": 'PkPrzem'},
                          {"stationCode": 249180010, "cityName": 'SlPszcz'},
                          {"stationCode": 249200180, "cityName": 'MpLimano'},
                          {"stationCode": 249220080, "cityName": 'PkSano'},
                          {"stationCode": 250140030, "cityName": 'DsBoga'},
                          {"stationCode": 250160130, "cityName": 'DsSzcz'},
                          {"stationCode": 250160360, "cityName": 'MpTar'},

                          {"stationCode": 250170390, "cityName": 'OpGlub'},
                          {"stationCode": 250180030, "cityName": 'OpOles'},
                          {"stationCode": 250190250, "cityName": 'DsZabk'},
                          {"stationCode": 250190530, "cityName": 'SlKato'},
                          {"stationCode": 250210050, "cityName": 'SkSw'},
                          {"stationCode": 251150250, "cityName": 'DsZgor'},
                          {"stationCode": 251160150, "cityName": 'DsPol'},
                          {"stationCode": 251180090, "cityName": 'LdSier'},
                          {"stationCode": 251200030, "cityName": 'LdSkier'},
                          {"stationCode": 251210040, "cityName": 'LbJarcz'},

                          {"stationCode": 251210120, "cityName": 'LbPula'},
                          {"stationCode": 252170110, "cityName": 'WpGnie'},
                          {"stationCode": 252200120, "cityName": 'MzLeg'},
                          {"stationCode": 253180150, "cityName": 'KpGrud'},
                          {"stationCode": 253180220, "cityName": 'KpBydg'},
                          {"stationCode": 254170140, "cityName": 'PmKos'},
                          {"stationCode": 254180060, "cityName": 'PmGdy'},
                          {"stationCode": 254180110, "cityName": 'PmGda'},
                          {"stationCode": 254220030, "cityName": 'WmGold'},
                          {"stationCode": 249180130, "cityName": 'SlCies'}]
    history_list = []
    for i in range(1, len(dictionaries_array) + 1):
        arr = dictionaries_array[slice(0, i)]
        df = prepareData.merge_climate_and_air_data(arr)
        df = df[df['pm10'].notna()]
        gru = notebooks.gru_regression(df.copy(), 128, 50, 'scale', i)
        history_list.append(gru)
    helpers.plot_willmott_for_cities(range(1, len(dictionaries_array) + 1), history_list)

    # compare_models(df)


def compare_models(df):
    notebooks.run_linear_regression_for_climate(df.copy(), 'models')
    notebooks.climate_tree_regression(df.copy(), 'models')
    history_list = []
    for neurons in [32, 64, 128, 216, 512, 1024]:
        rnn = notebooks.simple_rnn_regression(df.copy(), neurons, 50, 'models')
        lstm = notebooks.lstm_regression(df.copy(), neurons, 50, 'models')
        gru = notebooks.gru_regression(df.copy(), neurons, 50, 'models')

        rnn.label = f'Simple RNN with {neurons} neurons'
        lstm.label = f'LSTM with {neurons} neurons'
        gru.label = f'GRU with {neurons} neurons'
        history_list.extend([rnn, lstm, gru])
    helpers.plot_model_history(history_list)


main()
