import pandas as pd
import gzip
import datetime
import os
import numpy as np
import pickle

WINDOW_SIZE = 2880 #a week of information 7*24*60 -> for two days, 2880
LOOK_AHEAD = 2 # how much we want to predict, 2 hours
INSTRUMENT_OF_INTEREST = 'EURUSD'

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df


def add_missing_dates(data):
    time = data['DateTime']
    for i, t in enumerate (time):
        tt = datetime.datetime.strptime(t.split(' ')[1], '%H:%M:%S.%f')
        min = tt + datetime.timedelta(0,60)
        min2 = tt + datetime.timedelta(0, 120)
        if len(time) < (i+1) and min !=  datetime.datetime.strptime(time[i+1].split(' ')[1], '%H:%M:%S.%f') and min2 ==  datetime.datetime.strptime(time[i+1].split(' ')[1], '%H:%M:%S.%f'):
            input = [t.split(' ')[0]+min.strftime("%H:%M:%S.%f"), data['BidOpen'][i], data['BidHigh'][i], data['BidLow'][i], data['BidClose'][i+1], data['AskOpen'][i], data['AskHigh'][i], data['AskLow'][i], data['AskClose'][i+1]]
            insert_row(i+1, data, input)
    return data


def clean_dataset(data, instrument):
    data = data.rename(columns={"DateTime": "DateTime", "BidOpen": "BidOpen"+instrument, "BidHigh": "BidHigh"+instrument, "BidLow": "BidLow"+instrument,"BidClose": "BidClose"+instrument,
                         "AskOpen": "AskOpen"+instrument,"AskHigh": "AskHigh"+instrument, "AskLow": "AskLow"+instrument,"AskClose": "AskClose"+instrument})
    for column in data.columns:
        if column == 'DateTime':
            continue
        try:
            m = max(data[column])
        except:
            continue
        data[column] = data[column]/m
    return data


def windowded (instrument):
    frames = []
    for filename in os.listdir('./data/'):
        if instrument in filename:
            try:
                data = pd.read_csv(gzip.open('./data/'+filename, 'rb'))
            except:
                continue
            data = add_missing_dates(data)
            print(instrument,filename)

            frames.append(data)
            result = pd.concat(frames)
    result = clean_dataset(result, instrument)

    # result = pd.read_csv(gzip.open('AUDCAD_2012_1.csv.gz', 'rb'))
    return result


def get_labels(dict, instrument):
    l=[]
    for k in dict:
        df = dict[k]
        print('lengh:', k, len(df))
        df.set_index('DateTime', inplace=True)
        # df = df.drop(columns=['DateTime'])
        l.append(df)
    combined_features = pd.concat(l, axis=1, join='inner')
    print(len(combined_features.columns), len(combined_features))

    # features = result.drop(columns=['DateTime'])

    if instrument == 'EURUSD':
        for iii in range(10078, len(combined_features) - WINDOW_SIZE - LOOK_AHEAD):
            # window_features = features[i:i+WINDOW_SIZE]
            combined_window_features = combined_features[iii:(iii + WINDOW_SIZE)]
            f = combined_window_features.to_numpy(dtype=np.float32)
            temp1 = combined_features['BidOpen'+instrument][iii+WINDOW_SIZE+LOOK_AHEAD]
            temp2 = combined_features['AskClose'+instrument][iii+WINDOW_SIZE+LOOK_AHEAD]

            if combined_window_features['AskClose'+instrument][-1] < temp1:
                label = [1, 0, 0]  # buy
            elif combined_window_features['BidOpen'+instrument][-1] > temp2:
                label = [0, 0, 1]  # sell
            else:
                label = [0, 1, 0]  # do not do anything

            # arr = list({'feature':f, 'label':label}.items())
            np.save('D:\\Programming\\Windowded\\' + str(iii) + '_feature.npy', f)
            np.save('D:\\Programming\\Windowded\\LABEL\\' + str(iii) + '_feature.npy', np.array(label, dtype=np.float32))


if __name__ == '__main__':
    # instruments = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'EURAUD',
    #             'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPCHF', 'GBPJPY',
    #             'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
    #             'USDCAD', 'USDCHF', 'USDJPY']
    #
    # result = {}
    # for instrument in instruments:
    #     result[instrument] = windowded(instrument)
    #
    # save_obj(result, './res')

    result = load_obj('./res')
    get_labels(result, 'EURUSD')
