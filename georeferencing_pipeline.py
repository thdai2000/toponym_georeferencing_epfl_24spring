import os
from utils import *
import json
from transform import *
from geopy.distance import geodesic
import folium
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import result_reader as rr
from visualizer import *
import pandas as pd
import numpy as np
import os
from utils import *
import json
from transform import *
from geopy.distance import geodesic
import folium
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.stats import sem, t
from numpy import mean
from collections import Counter
from pyproj import Transformer
import math
import subprocess
import time
from tqdm import tqdm
import glob
from geopy.geocoders import GoogleV3

GOOGLE_V3_API_KEY = "[YOUR KEY]"
OPENAI_API = "[YOUR KEY]"
OPENAI_SERVER = "[YOUR SERVER]"


if __name__ == '__main__':
    RESULTS_DIR = './results3'
    map_names = os.listdir(RESULTS_DIR)
    TRANSFORM_ORDER = 1

    for MAP_ID in tqdm(map_names):

        if 'georeferencing_data.pkl' in os.listdir('{}/{}'.format(RESULTS_DIR, MAP_ID)):
            continue

        if '12148_' in MAP_ID:
            CITY = 'Paris'
        else:
            CITY = 'Jerusalem'

        print("Processing map {}...".format(MAP_ID))
        # MAP_PATH = glob.glob('{}/{}/{}.jp{}g'.format(RESULTS_DIR,MAP_ID, MAP_ID, '*'))[0]
        TOPONYM_POLYGONS = glob.glob('{}/{}/toponym_detections.json'.format(RESULTS_DIR, MAP_ID))[0]

        try:
            CONTROL_PTS = glob.glob('{}/{}/{}'.format(RESULTS_DIR, MAP_ID, '*.points'))[0]
        except:
            CONTROL_PTS = './data/control_points/{}/{}.jpg.points'.format(CITY.lower(), MAP_ID)

        toponyms = rr.read_json_nested(TOPONYM_POLYGONS)

        # denoise toponyms
        toponyms_denoised = []
        for tp in toponyms:
            words = tp['text'].split(' ')
            if len(words) >= 2 and any([len(w) >= 2 for w in words]):
                toponyms_denoised.append(tp)

        # # plot denoised toponyms
        # vis = PolygonVisualizer()
        # vis.canvas_from_image(Image.open(MAP_PATH))
        # vis.draw_toponyms(toponyms_denoised).save('{}/{}/toponym_detections_denoised.jpg'.format(RESULTS_DIR, MAP_ID))
        print("Done denoising.")

        df_toponyms = convert_json_to_df(toponyms_denoised)

        # normalization
        def ask_gpt(prompt):

            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            data_json = json.dumps(data)
            curl_command = [
                'curl',
                '-X', 'POST',
                OPENAI_SERVER,
                '-H', 'Authorization: Bearer {}'.format(OPENAI_API),
                '-H', 'Content-Type: application/json',
                '-d', data_json
            ]
            result = subprocess.run(curl_command, capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:
                try:
                    answer = json.loads(result.stdout)['choices'][0]['message']['content']
                    return answer
                except:
                    return 'Unclear.'
            else:
                print(f"Curl command failed with return code {result.returncode}")
                print(result.stderr)
                return None

        print("Start normalizing...")
        normalized_names = []
        for name in tqdm(df_toponyms['text'].values.tolist()):

            question = "This is a toponym extracted from an old map of {}, please normalize it into its most possible full name that is recognizable by the GoogleV3 Geocoder: {}. Please only respond with the normalized name; if it's unclear, simply return 'Unclear'.".format(
                CITY, name)

            while (True):
                try:
                    answer = ask_gpt(question)
                    if answer is not None:
                        break
                    else:
                        time.sleep(10)
                        continue
                except Exception as e:
                    continue

            normalized_names.append(answer)

        df_toponyms['normalized_name'] = normalized_names
        df_toponyms = df_toponyms[
            df_toponyms['normalized_name'].apply(lambda x: 'Unclear' not in x)]
        print('Done normalizing.')

        # geocoding
        geolocator = GoogleV3(api_key=GOOGLE_V3_API_KEY)

        # geocoding
        def get_long_lat(x):
            location = geolocator.geocode(x, components={"city": CITY})
            if location:
                return (location.longitude, location.latitude)
            else:
                return None

        print('Start geocoding...')
        df_toponyms['geocoded_coordinates'] = df_toponyms['normalized_name'].apply(get_long_lat)
        df_toponyms = df_toponyms.dropna(subset=['geocoded_coordinates'])  # drop NaN results
        df_toponyms = df_toponyms[df_toponyms['geocoded_coordinates'].apply(
            lambda x: within_boundary(x[0], x[1], CITY))]  # drop unreliable results

        df_toponyms.to_pickle('{}/{}/georeferencing_data.pkl'.format(RESULTS_DIR, MAP_ID))

        # df_toponyms = pd.read_pickle('{}/{}/georeferencing_data.pkl'.format(RESULTS_DIR, MAP_ID))

        if len(df_toponyms) <= 3:
            print('Not enough points! Failed {}.'.format(MAP_ID))
            continue

        # save georeferencing data
        print('Done geocoding. Remaining {} toponyms.'.format(str(len(df_toponyms))))

        # georeferencing
        print('Start georeferencing...')
        df_control_points = read_control_points(CONTROL_PTS)
        df_control_points.sourceY *= -1

        pixel_x_sample = df_toponyms["centroid_x"].tolist()
        pixel_y_sample = df_toponyms["centroid_y"].tolist()
        longitude_sample = df_toponyms['geocoded_coordinates'].apply(lambda x: x[0]).tolist()
        latitude_sample = df_toponyms['geocoded_coordinates'].apply(lambda x: x[1]).tolist()

        pixel_x_gt = df_control_points['sourceX']
        pixel_y_gt = df_control_points['sourceY']
        longitude_gt = df_control_points['mapX']
        latitude_gt = df_control_points['mapY']

        # stack x and y to 2-dim sample
        X_sample = np.column_stack([pixel_x_sample, pixel_y_sample])
        y_sample = np.column_stack([longitude_sample, latitude_sample])
        X_gt = np.column_stack([pixel_x_gt, pixel_y_gt])
        y_gt = np.column_stack([longitude_gt, latitude_gt])

        # get georeferencing results
        # Try different residual_threshold values
        transform_order = TRANSFORM_ORDER
        thresholds = np.linspace(0.001, 0.02, 20)
        mean_distances = []
        conf_intervals = []
        all_distances = []
        inlier_toponyms = Counter()
        outlier_toponyms = Counter()
        inlier_area = Counter()
        outlier_area = Counter()
        thresholds_valid = []

        for threshold in tqdm(thresholds):

            accumulated_distances = []
            # repete 5 times for each threshold
            for i in range(5):
                # Apply RANSAC
                # use polynomial transform, equivalent to multi-output linear regression
                # affine transform when degree is 1
                distances = []
                try:
                    ransac = RANSACRegressor(
                        estimator=make_pipeline(PolynomialFeatures(degree=transform_order, include_bias=False), LinearRegression()),
                        min_samples=5,
                        residual_threshold=threshold,
                        max_trials=1000)
                    ransac.fit(X_sample, y_sample)
                    y_pred = ransac.predict(X_gt)

                    for i in range(len(X_gt)):
                        if CITY == 'Jerusalem':
                            lon_pred, lat_pred = transform_cs(y_pred[i, 0], y_pred[i, 1], city=CITY)
                            lon_gt, lat_gt = y_gt[i, 0], y_gt[i, 1]
                        else:
                            lon_pred, lat_pred = transform_cs(y_pred[i, 0], y_pred[i, 1], city=CITY)
                            lon_gt, lat_gt = transform_cs(y_gt[i, 0], y_gt[i, 1], city=CITY)
                        distance = euclidean(lon_pred, lat_pred, lon_gt, lat_gt)
                        distances.append(distance)
                except:
                    continue
                accumulated_distances += distances

            # calculate the error bar
            if len(accumulated_distances) >= 2:
                mean_dist = mean(accumulated_distances)
                std_err = sem(accumulated_distances)
                h = std_err * t.ppf((1 + 0.95) / 2., len(accumulated_distances) - 1)

                all_distances.append(accumulated_distances)
                mean_distances.append(mean_dist)
                conf_intervals.append(h)
                thresholds_valid.append(threshold)
            else:
                continue

        # Plotting
        if len(thresholds_valid) > 0:
            plt.figure(figsize=(10, 5))
            plt.errorbar(thresholds_valid, mean_distances, yerr=conf_intervals, fmt='-o')
            plt.title('Mean Distance with 95% Confidence Intervals for Different Residual Thresholds')
            plt.xlabel('Residual Threshold')
            plt.ylabel('Mean Distance (meters)')
            plt.grid(True)
            plt.savefig('{}/{}/georeferencing_results.png'.format(RESULTS_DIR, MAP_ID))
            # plt.show()

            georeferencing_output = pd.DataFrame({'threshold': thresholds_valid,
                                                  'mean_distance': mean_distances,
                                                  'confidence_interval': conf_intervals,
                                                  'distances': all_distances})
            georeferencing_output.to_csv('{}/{}/georeferencing_results_order{}.csv'.format(RESULTS_DIR, MAP_ID, transform_order), index=False)

        else:
            continue

        print('Done georeferencing.')
