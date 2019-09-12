
import firebase_admin
import pandas as pd
from firebase_admin import credentials, db
import numpy as np
from scipy import stats
import time
import pickle
import sklearn
from sklearn.externals import joblib

# feature extraction
def get_mean(l, col):
    """mean of accelerometer data"""
    return np.mean(l[col])

def get_median(l, col):
    """median of accelerometer data"""
    return np.median(l[col])


def get_mode(l, col):
    """Mode of accelerometer data"""
    return stats.mode(l[col])[0][0]

def no_of_events(l,col):
    """number of times a touchdown event occured"""
    count = 0
    for item in l[col]:
        if not np.isnan(item):
            count+=1
    return count

def no_of_true_events(l,col):
    """number of times the sensor has a true value"""
    count = 0
    for item in l[col]:
        if not np.isnan(item):
            if(item):
                count+=1
    return count

def duration_between_consecutive_events(l,col):
    prev_timestamp = 0
    current_timestamp = 0
    difference = []
    """time duration between two steps of """
    for index,item in enumerate(l[col]):
        if not np.isnan(item):
            if(item):
                current_timestamp = l['timestamp'].iloc[index]
                difference.append(current_timestamp-prev_timestamp)
                prev_timestamp = current_timestamp
                
    if(difference):
        difference.pop(0)
        return difference
    else:
        return float('nan')
    
def get_Time_difference_mean(l):
    """mean of time difference between two consecutive steps"""
    return np.mean(l)

def get_Time_difference_median(l):
    """median of time difference between two consecutive steps"""
    return np.nanmedian(l)

def feature_extraction(data_df):
    feature_df = pd.DataFrame(columns = ['Mean_Ax','Mode_Ax', 'Median_Ax', 'Mean_Ay','Mode_Ay','Median_Ay','Mean_Az','Mode_Az','Median_Az',
                                         'Mean_Bx','Mode_Bx', 'Median_Bx', 'Mean_By','Mode_By','Median_By','Mean_Bz','Mode_Bz','Median_Bz',
                                         'Mean_Cx','Mode_Cx', 'Median_Cx', 'Mean_Cy','Mode_Cy','Median_Cy','Mean_Cz','Mode_Cz','Median_Cz',
                                         'no_of_events_occured','no_of_True_events_occured_A','no_of_True_events_occured_B',
                                         'no_of_True_events_occured_C','no_of_True_events_occured_D',
                                         'Mean_impact','Mode_impact','Median_impact','Time_difference_mean_A','Time_difference_median_A', 'Time_difference_mean_B','Time_difference_median_B',
                                        'Time_difference_mean_C','Time_difference_median_C', 'Time_difference_mean_D','Time_difference_median_D'])

    td_mn_A = get_Time_difference_mean(duration_between_consecutive_events(data_df,'A'))
    td_mn_B = get_Time_difference_mean(duration_between_consecutive_events(data_df,'B'))
    td_mn_C = get_Time_difference_mean(duration_between_consecutive_events(data_df,'C'))
    td_mn_D = get_Time_difference_mean(duration_between_consecutive_events(data_df,'D'))
    td_md_A = get_Time_difference_median(duration_between_consecutive_events(data_df,'A'))
    td_md_B = get_Time_difference_median(duration_between_consecutive_events(data_df,'B'))
    td_md_C = get_Time_difference_median(duration_between_consecutive_events(data_df,'C'))
    td_md_D = get_Time_difference_median(duration_between_consecutive_events(data_df,'D'))

    feature_df = feature_df.append({'Mean_Ax':get_mean(data_df ,'Ax'),'Mode_Ax':get_mode(data_df ,'Ax'),'Median_Ax':get_median(data_df,'Ax'),
                                    'Mean_Ay':get_mean(data_df ,'Ay'),'Mode_Ay':get_mode(data_df ,'Ay'),'Median_Ay':get_median(data_df ,'Ay'),
                                    'Mean_Az':get_mean(data_df ,'Az'),'Mode_Az':get_mode(data_df ,'Az'),'Median_Az':get_median(data_df ,'Az'),
                                    'Mean_Bx':get_mean(data_df ,'Bx'),'Mode_Bx':get_mode(data_df ,'Bx'),'Median_Bx':get_median(data_df ,'Bx'), 
                                    'Mean_By':get_mean(data_df ,'By'),'Mode_By':get_mode(data_df ,'By'),'Median_By':get_median(data_df ,'By'),
                                    'Mean_Bz':get_mean(data_df ,'Bz'),'Mode_Bz':get_mean(data_df ,'Bz'),'Median_Bz':get_median(data_df ,'Bz'),
                                    'Mean_Cx':get_mean(data_df ,'Cx'),'Mode_Cx':get_mode(data_df ,'Cx'),'Median_Cx':get_median(data_df ,'Cx'),
                                    'Mean_Cy':get_mean(data_df ,'Cy'),'Mode_Cy':get_mode(data_df ,'Cy'),'Median_Cy':get_median(data_df ,'Cy'),
                                    'Mean_Cz':get_mean(data_df ,'Cz'),'Mode_Cz':get_mode(data_df ,'Cz'),'Median_Cz':get_median(data_df ,'Cz'),
                                    'no_of_events_occured':no_of_events(data_df ,'A'),
                                    'no_of_True_events_occured_A':no_of_true_events(data_df ,'A'),
                                    'no_of_True_events_occured_B':no_of_true_events(data_df ,'B'),
                                    'no_of_True_events_occured_C':no_of_true_events(data_df ,'C'),
                                    'no_of_True_events_occured_D':no_of_true_events(data_df ,'D'),
                                    'Mean_impact':get_mean(data_df ,'impact'),'Mode_impact':get_mode(data_df ,'impact'),'Median_impact':get_median(data_df ,'impact'),
                                    'Time_difference_mean_A':td_mn_A,'Time_difference_median_A':td_md_A, 
                                    'Time_difference_mean_B':td_mn_B,'Time_difference_median_B':td_md_B,
                                    'Time_difference_mean_C':td_mn_C,'Time_difference_median_C':td_md_C, 
                                    'Time_difference_mean_D':td_mn_D,'Time_difference_median_D':td_md_D},ignore_index=True)
    return feature_df

def transpose_data(data):
    data = pd.DataFrame(data).T
    data = data.reset_index(level=0).rename(columns = {'index' : 'timestamp'})
    data['timestamp'].astype('int64')
    data = data.sort_values('timestamp', ascending=False)
    return data 

def clean_data(data):
    accelerometer_data = transpose_data(data['AccelerometerData'])
    if 'SensorReading' in data:
        sensor_data = transpose_data(data['SensorReading'])
    else:
        sensor_data = pd.DataFrame(columns=['timestamp','A','B','C','D','forefoot','heel','impact','varus','sole'])
    sensor_data[['A','B','C','D']]*= 1
    sensor_data = pd.merge(accelerometer_data, sensor_data, how='outer', on=['timestamp', 'timestamp'])
    sensor_data = sensor_data.drop(['forefoot', 'heel','sole','varus' ], axis=1)
    sensor_data = sensor_data.reset_index(drop=True).rename(columns={'index':'timestamp'})
    return sensor_data

def change_event_handler(event):
    left_sole = db.reference('Insole/Left/DeviceName').get()
    print('here')
    while True:
        print(f"data event observed polling every {WAIT_DURATION} seconds")
        time.sleep(WAIT_DURATION)
        data = db.reference(f'Data/{left_sole}').get()
        if data:
            clean_data_df = clean_data(data)
            feature_df = feature_extraction(clean_data_df)
            feature_df = feature_df.fillna(-1000)
            prediction = EVENT_MAPPING[model.predict(feature_df)[0]]
            print(prediction)
            db.reference('Predictions').child(str(int(time.time()))).set(prediction)
            db.reference('Data').delete()
        else:
            print("shoe activity ended")
            print("waiting for shoe event")
            break

if __name__ == "__main__":
    WAIT_DURATION = 5
    EVENT_MAPPING = {1:'walking', 2:'running', 3:'sitting', 4:'standing', 5:'stairup', 6:'stairdown', 7:'elliptical'}
    with open(r"saved_decision_tree_model_latest.pkl", "rb") as input_file:
        model = joblib.load(input_file)

    cred = credentials.Certificate('myawesomeinsole-firebase-adminsdk.json')
    firebase_admin.initialize_app(cred, {'databaseURL' : 'https://myawesomeinsole.firebaseio.com'}) 
    # print("waiting for shoe event")      
    # root = db.reference('Data').listen(change_event_handler)
    left_sole = db.reference('Insole/Left/DeviceName').get()

    print(f"polling every {WAIT_DURATION} seconds")
    counter = 0
    while True:
        time.sleep(WAIT_DURATION)
        counter+=1
        data = db.reference(f'Data/{left_sole}').get()
        if data:
            clean_data_df = clean_data(data)
            feature_df = feature_extraction(clean_data_df)
            feature_df = feature_df.fillna(-1000)
            prediction = EVENT_MAPPING[model.predict(feature_df)[0]]
            print(prediction)
            db.reference('Predictions').child(str(int(time.time()))).set(prediction)
            db.reference('Data').delete()
        else:
            if counter==5:
                print("No shoe activity")
                counter=0
