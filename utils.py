from PIL import Image, ImageDraw
import pandas as pd
from io import StringIO
import os
import json
from pyproj import Transformer
import math
import numpy as np


def visualize_point(img_path, point_x, point_y):

    with Image.open(img_path) as img:
        # Create an ImageDraw object to draw on the image
        draw = ImageDraw.Draw(img)

        # Coordinates of the point you want to draw
        # (x, y) format, (0, 0) is the top-left corner of the image

        # Draw the point on the image
        # The 'fill' parameter is the color of the point
        # 'width' parameter specifies the circle's diameter to represent the point
        draw.ellipse((point_x - 10, point_y - 10, point_x + 10, point_y + 10), fill='red', width=1)

        # Display the image
        img.show()

    return img


def read_control_points(file_path):

    with open(file_path, 'r') as file:
        content = file.read()

    # Since the actual data seems to start after the CRS definition, let's split the content to ignore the CRS part and focus on the data.
    data_part = content.split('\n', 1)[1]  # Skip the first line containing the CRS definition

    # Convert the data part to a DataFrame for analysis
    data_io = StringIO(data_part)
    df = pd.read_csv(data_io, sep=",").dropna()

    return df


def visualize_control_points(img_path, control_points_path):

    df_control_points = read_control_points(control_points_path)

    with Image.open(img_path) as img:
        # Create an ImageDraw object to draw on the image
        draw = ImageDraw.Draw(img)

        for i in range(len(df_control_points)):
            point_x = df_control_points.iloc[i]['sourceX']
            point_y = -df_control_points.iloc[i]['sourceY']
            draw.ellipse((point_x - 10, point_y - 10, point_x + 10, point_y + 10), fill='red', width=1)

        img.show()

    return img


def draw_points(img, points, color='red', show=True):
    '''

    :param img: pillow img
    :param points: ([x1,x2,x3,...],[y1,y2,y3,...])
    :return:
    '''

    # Create an ImageDraw object to draw on the image
    draw = ImageDraw.Draw(img)

    for i in range(len(points[0])):
        point_x = points[0][i]
        point_y = points[1][i]
        draw.ellipse((point_x - 10, point_y - 10, point_x + 10, point_y + 10), fill=color, width=5)

    if show:
        img.show()

    return img


def visualize_polygons(img_path, control_points_path):


    return


def polygon_area(x_coords, y_coords):
    """Calculate the area of a polygon given separate lists of x and y coordinates."""
    area = 0.0
    n = len(x_coords)
    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j]
        area -= x_coords[j] * y_coords[i]
    area = abs(area) / 2.0
    return area

def polygon_centroid(x_coords, y_coords):
    """Calculate the centroid of a polygon given separate lists of x and y coordinates."""
    area = polygon_area(x_coords, y_coords)
    cx = 0.0
    cy = 0.0
    n = len(x_coords)
    for i in range(n):
        j = (i + 1) % n
        common = x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i]
        cx += (x_coords[i] + x_coords[j]) * common
        cy += (y_coords[i] + y_coords[j]) * common
    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    return (-cx, -cy)


def read_polygons_vgg(file_path):
    df_toponyms = pd.read_csv(file_path)
    df_toponyms['map_id'] = df_toponyms['filename'].apply(lambda x: x.split('.')[0])

    def load_toponym(x):
        attributes_dict = json.loads(x)
        if 'toponym' in attributes_dict.keys():
            return attributes_dict['toponym']
        else:
            return ''
    df_toponyms['extracted_toponym'] = df_toponyms['region_attributes'].apply(load_toponym)
    # df_toponyms['toponym'] = df_toponyms['region_attributes'].apply(lambda x: json.loads(x)['identifier'])
    df_toponyms['polygon_x'] = df_toponyms['region_shape_attributes'].apply(lambda x: json.loads(x)['all_points_x'])
    df_toponyms['polygon_y'] = df_toponyms['region_shape_attributes'].apply(lambda x: json.loads(x)['all_points_y'])

    df_toponyms = df_toponyms[['map_id', 'extracted_toponym', 'polygon_x', 'polygon_y']]
    df_toponyms['polygon_centroid_x'] = df_toponyms.apply(
        lambda row: polygon_centroid(row['polygon_x'], row['polygon_y'])[0], axis=1)
    df_toponyms['polygon_centroid_y'] = df_toponyms.apply(
        lambda row: polygon_centroid(row['polygon_x'], row['polygon_y'])[1], axis=1)
    return df_toponyms


def convert_json_to_df(toponyms):

    text = []
    polygon_centroid_x = []
    polygon_centroid_y = []

    for tp in toponyms:
        text.append(tp['text'])
        polygon_centroid_x.append(tp['center'][0])
        polygon_centroid_y.append(tp['center'][1])

    df_toponyms = pd.DataFrame({'text': text, 'centroid_x': polygon_centroid_x, 'centroid_y': polygon_centroid_y})

    return df_toponyms

def read_control_points_as_df(dir_path):
    df_control_points = None
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        df_tmp = read_control_points(file_path)
        map_id = file_name.split('.')[0]
        df_cols = df_tmp.columns.tolist()
        df_tmp['map_id'] = [map_id] * len(df_tmp)
        df_tmp = df_tmp[['map_id'] + df_cols]
        df_control_points = pd.concat((df_control_points, df_tmp), ignore_index=True)
    return df_control_points


def read_toponym_coord(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dict_toponym = json.load(file)
        df_toponym_coord = pd.DataFrame(dict_toponym).T.reset_index()
        df_toponym_coord.columns = ['toponym', 'latitude', 'longitude']
    return df_toponym_coord


from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = sin(delta_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000

    return distance


def within_boundary(lon, lat, city):

    local_boundary = {
        'Paris': {
            'x_min': 1.443,
            'x_max': 3.563,
            'y_min': 48.140,
            'y_max': 49.050
        },
        'Jerusalem': {
            'x_min': 35.00,
            'x_max': 35.40,
            'y_min': 31.60,
            'y_max': 32.00
        }
    }

    if lon < local_boundary[city]['x_max'] \
        and lon > local_boundary[city]['x_min'] \
        and lat < local_boundary[city]['y_max'] \
        and lat > local_boundary[city]['y_min']:
        return True
    else:
        return False


def transform_cs(lon, lat, city):

    local_system = {
        'Paris': "EPSG:27561",  # NTF (Paris) / Lambert zone I
        'Jerusalem': "EPSG:28193"  # Palestine 1923 / Israeli CS Grid
    }

    transformer = Transformer.from_crs("EPSG:4326", local_system[city], always_xy=True)

    x, y = transformer.transform(lon, lat)

    return x, y


def euclidean(x1, y1, x2, y2):

    # Calculate the Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance
