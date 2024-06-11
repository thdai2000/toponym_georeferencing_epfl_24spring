import os
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from english_encoding import *
from bezier_utils import *
import random
from PIL import Image, ImageDraw, ImageEnhance

class GrouperDataset:
    def __init__(self, image_root_folder) -> None:
        '''
        images: {image_id: {'file_name': str, 'id': int, 'width': int, 'height': int}}
        annos: {image_id: [{'rec': [[int,]], 'bezier': [[float,]]}]}

        'rec' is a 35-element list of integers representing the text in the polygon, for example:
        self.annos[image_id][toponym_id]['rec'][word_id] = [char_id1, char_id2, char_id3, char_id4, char_id5, char_id6, char_id7, char_id8, ...]

        'bezier' is a list of 16 floats representing 8 control points of the bezier curve
        for example:
        self.annos[image_id][toponym_id]['bezier'][word_id] = [
            upper_x1, upper_y1,
            upper_x2, upper_y2,
            upper_x3, upper_y3,
            upper_x4, upper_y4,

            lower_x1, lower_y1,
            lower_x2, lower_y2,
            lower_x3, lower_y3,
            lower_x4, lower_y4
        ]
        '''
        self.images = {}
        self.annos = {}
        self.words = {}
        self.image_root_folder = image_root_folder

    def register_image(self, image_name):
        # Check if the image is already registered
        for image in self.images.values():
            if image['file_name'] == image_name:
                print("Image already registered")
                return None
            
        # Check file exists
        image_path = os.path.join(self.image_root_folder, image_name)
        if not os.path.exists(image_path):
            print("Image file not found")
            return None
        
        # Check width and height
        image = Image.open(image_path)
        width, height = image.size

        # Register the image
        image_id = len(self.images)
        image_record = {
            "file_name": image_name, 
            "id": image_id, 
            "width": width, 
            "height": height,
            }
        self.images[image_id] = image_record
        self.annos[image_id] = []
        self.words[image_id] = []

        return image_id
    
    def _approx_bezier(self, poly_upper_x, poly_upper_y, poly_lower_x, poly_lower_y, max_error_ratio = 0.5, max_attempts = 10):
        import bezier_utils as butils
        
        success = False
        attempt = 0
        cpts_upper_x, cpts_upper_y = [], []
        cpts_lower_x, cpts_lower_y = [], []
        while not success and attempt < max_attempts:
            success = True
            error_ratio = max_error_ratio/max_attempts*(attempt + 1)
            poly_length_upper = butils.polyline_length(poly_upper_x, poly_upper_y)
            poly_length_lower = butils.polyline_length(poly_lower_x, poly_lower_y)
            avg_length = (poly_length_upper + poly_length_lower) / 2
            max_error = avg_length * error_ratio

            cpts_upper_x, cpts_upper_y = butils.bezier_from_polyline(poly_upper_x, poly_upper_y, max_error)

            cpts_lower_x, cpts_lower_y = butils.bezier_from_polyline(poly_lower_x, poly_lower_y, max_error)

            bezier_length_upper = butils.bezier_length(cpts_upper_x, cpts_upper_y)
            bezier_length_lower = butils.bezier_length(cpts_lower_x, cpts_lower_y)

            # Check if two bezier curves have similar length, if not, raise an error
            thresh = 1.2
            ratio_upper = poly_length_upper/bezier_length_upper
            ratio_lower = poly_length_lower/bezier_length_lower
            if ratio_upper > thresh or ratio_upper < 1/thresh or ratio_lower > thresh or ratio_lower < 1/thresh:
                success = False
                attempt += 1

        return success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y

    def register_annotation(self, image_id, text_polygon_dict):
        import bezier_utils as butils
        if image_id not in self.images:
            print("Image not found")
            return None
        
        if not text_polygon_dict:
            print("Empty annotation")
            return None
        
        group = {'rec':[], 'bezier':[]}
        for rec, polygon in zip(text_polygon_dict['rec'], text_polygon_dict['polygon']):
            upper_half = polygon[:8]
            lower_half = list(reversed(polygon[8:]))

            center = (sum([p[0] for p in upper_half + lower_half]) / 16, sum([p[1] for p in upper_half + lower_half]) / 16)

            poly_upper_x, poly_upper_y = [p[0] for p in upper_half], [p[1] for p in upper_half]
            poly_lower_x, poly_lower_y = [p[0] for p in lower_half], [p[1] for p in lower_half]

            success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y = self._approx_bezier(poly_upper_x, poly_upper_y, poly_lower_x, poly_lower_y)

            if not success:
                print("Failed to approximate bezier curve")
                raise Exception("Failed to approximate bezier curve")

            # bezier_pts = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
            bezier_pts = []
            for i in range(4):
                bezier_pts.append(round(cpts_upper_x[i], 2))
                bezier_pts.append(round(cpts_upper_y[i], 2))
            for i in range(4):
                bezier_pts.append(round(cpts_lower_x[i], 2))
                bezier_pts.append(round(cpts_lower_y[i], 2))

            group['rec'].append(rec)
            group['bezier'].append(bezier_pts)
            self.words[image_id].append({'rec': rec, 'bezier': bezier_pts, 'center': center, 'group_id': len(self.annos[image_id]), 'id_in_group': len(group['rec']) - 1})

        self.annos[image_id].append(group)
        return True
    
    def gen_font_embed(self, image_id, deepfont_encoder):
        import DeepFont
        import bezier_utils as butils

        net = deepfont_encoder

        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = Image.open(image_path)

        words = self.words[image_id]

        # Get word snippets from the image according to the annotation
        snippets = []

        for word in words:
            bezier_pts = word['bezier']
            snippet = butils.get_bezier_bbox(image, bezier_pts)
            snippets.append(snippet)

        # Encode the snippets
        features = DeepFont.EncodeFontBatch(net, snippets)

        self.words[image_id] = [{**word, 'font_embed': feature} for word, feature in zip(words, features)]

    def gen_font_embed_all(self, deepfont_encoder_path, device='cpu'):
        '''
            One minute inference for all 200 full map patches in rumsey dataset
        '''
        from tqdm import tqdm
        import DeepFont

        net = DeepFont.load_model(deepfont_encoder_path, device)

        for image_id in tqdm(self.images):
            self.gen_font_embed(image_id, net)

    def save_annotations_to_file(self, file_path):
        with open(file_path, 'w') as f:
            f.write(json.dumps({'images': self.images, 'annotations': self.annos, 'words': self.words}))

    def load_annotations_from_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
            self.images = {int(key):value for key, value in data['images'].items()}
            self.annos = {int(key):value for key, value in data['annotations'].items()}
            self.words = {int(key):value for key, value in data['words'].items()}

    def draw_annotations(self, image_id, decode_text):
        import bezier_utils as butils
        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = plt.imread(image_path)
        plt.imshow(image)
        for group in self.annos[image_id]:
            # Assign random color to each group
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            avg_xs = []
            avg_ys = []
            for rec, bezier_pts in zip(group['rec'], group['bezier']):
                upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
                lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

                upper_half_x = [p[0] for p in upper_half]
                upper_half_y = [p[1] for p in upper_half]
                lower_half_x = [p[0] for p in lower_half]
                lower_half_y = [p[1] for p in lower_half]

                # Convert bezier curve to polyline
                upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
                lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

                # Plot begin points
                plt.scatter(upper_half_x[0], upper_half_y[0], color='black')
                plt.scatter(lower_half_x[0], lower_half_y[0], color='black')

                plt.plot(upper_half_x, upper_half_y, color=color)
                plt.plot(lower_half_x, lower_half_y, color=color)
                text = decode_text(rec)
                plt.text(upper_half[0][0], upper_half[0][1], text, color=color)

                avg_x = (upper_half_x[0] + upper_half_x[-1] + lower_half_x[0] + lower_half_x[-1]) / 4
                avg_y = (upper_half_y[0] + upper_half_y[-1] + lower_half_y[0] + lower_half_y[-1]) / 4

                avg_xs.append(avg_x)
                avg_ys.append(avg_y)

            avg_xs = avg_xs
            avg_ys = avg_ys
            plt.plot(avg_xs, avg_ys, color='red')
            plt.scatter(avg_xs[0], avg_ys[0], color='black')

        plt.show()

    def sample(self, image_id, sample_count, closest_pts_count = 15, non_overlap = False):
        import bezier_utils as butils
        samples = []
        words = self.words[image_id]

        if sample_count > len(words):
            sample_count = len(words)

        word_samples = np.random.choice(words, sample_count, replace=not non_overlap)

        def _anchors(_word, type = '111222333'):
            arr = [
                ((_word['bezier'][0] + _word['bezier'][14])/2, (_word['bezier'][1] + _word['bezier'][15])/2), 
                ((_word['bezier'][6] + _word['bezier'][8])/2, (_word['bezier'][7] + _word['bezier'][9])/2),
                _word['center']
            ]
            if type == '111222333':
                return [arr[0], arr[0], arr[0], arr[1], arr[1], arr[1], arr[2], arr[2], arr[2]]
            elif type == '123123123':
                return [arr[0], arr[1], arr[2], arr[0], arr[1], arr[2], arr[0], arr[1], arr[2]]
        
        for word in word_samples:
            word_anchors = _anchors(word, type='111222333')
            _, nabb_l, nabb_s = butils.get_bezier_bbox_params(word['bezier'])

            nabb_l_new = nabb_l * np.linalg.norm(nabb_s) / np.linalg.norm(nabb_l)

            import cv2
            mat_src = np.float32([np.array(word['center']),np.array(word['center']) + nabb_l,np.array(word['center']) + nabb_s])  
            mat_dst = np.float32([np.array(word['center']),np.array(word['center']) + nabb_l_new,np.array(word['center']) + nabb_s])
            T = cv2.getAffineTransform(mat_src, mat_dst)
            

            anchors_to_compare = []
            dictionary_candidates = []
            inliners = []
            for w in words:
                if w['group_id'] == word['group_id']:
                    inliners.append(w)
                anchors_to_compare.append(_anchors(w, type='123123123'))
                dictionary_candidates.append(w)

            dist_vectors = np.array(anchors_to_compare) - np.array(word_anchors)
            dist_vectors_transformed = np.einsum('ji,akj->aki', T[:,:2], dist_vectors)
            distances = np.min(np.linalg.norm(dist_vectors_transformed, axis=2), axis=1)

            closest_indices = np.argsort(distances)[:closest_pts_count + 1]

            dictionary = []
            for i in closest_indices:
                dictionary.append(dictionary_candidates[i])

            complete = True
            for w in inliners:
                if w not in dictionary:
                    # Remove the word from the inliners
                    inliners.remove(w)
                    complete = False

            '''
                word: the word to be sampled around
                dictionary: the closest set of words to the word (non-inclusive of the word itself)
                toponym: the words that are in the same toponym as the word (preserve the order)
                complete: whether the dictionary contains all words in topynym, or, whether the toponym is complete in this scope
            '''
            samples.append({'word': word, 'dictionary': dictionary[1:], 'toponym': inliners, 'complete': complete})

        return samples

    def sample_ratio(self, image_id, sample_ratio, closest_pts_count = 15, non_overlap = False):
        sample_count = int(len(self.words[image_id]) * sample_ratio)
        return self.sample(image_id, sample_count, closest_pts_count, non_overlap)    

    def draw_sample(self, image_id, sample, decode_text):
        '''
            sample = {'word': word, 'dictionary': dictionary, 'toponym': toponym, 'complete': complete}
        '''
        words = self.words[image_id]

        import bezier_utils as butils
        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = plt.imread(image_path)
        plt.imshow(image)

        # Draw the sample
        word = sample['word']
        group_id = word['group_id']
        dictionary = sample['dictionary']
        toponym = sample['toponym']
        use_font_embed = False

        # Draw all words
        for aword in words:
            bezier_pts = aword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            plt.plot(upper_half_x, upper_half_y, color='grey')
            plt.plot(lower_half_x, lower_half_y, color='grey')
            text = decode_text(aword['rec'])
            aword_group_id = aword['group_id']
            if aword_group_id == group_id:
                plt.text(upper_half[0][0], upper_half[0][1], text, color='magenta')
            else:
                plt.text(upper_half[0][0], upper_half[0][1], text, color='grey')


        if 'font_embed' in word:
            word_font_embed = word['font_embed']
            use_font_embed = True

        for dword in dictionary:
            bezier_pts = dword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            similarity_str = ''
            if use_font_embed:
                dword_font_embed = dword['font_embed']
                euclidean_distance = np.linalg.norm(np.array(word_font_embed) - np.array(dword_font_embed))
                similarity_str = f':{euclidean_distance:.2f}'

            plt.plot(upper_half_x, upper_half_y, color='red')
            plt.plot(lower_half_x, lower_half_y, color='red')
            text = decode_text(dword['rec'])
            plt.text(upper_half[0][0], upper_half[0][1], text + similarity_str, color='red')

        for id, tword in enumerate(toponym):
            bezier_pts = tword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            plt.plot(upper_half_x, upper_half_y, color='blue')
            plt.plot(lower_half_x, lower_half_y, color='blue')
            text = decode_text(tword['rec'])
            plt.text(lower_half[0][0], lower_half[0][1], text + f': {id}', color='blue')

        for wword in [word]:
            bezier_pts = wword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            plt.plot(upper_half_x, upper_half_y, color='yellow')
            plt.plot(lower_half_x, lower_half_y, color='yellow')
            text = decode_text(wword['rec'])
            plt.text(upper_half[0][0], upper_half[0][1], text, color='yellow')
        
        if sample['complete']:
            plt.title('Complete')
        else:
            plt.title('Incomplete')
        plt.show()

    def bezier_centralized(self, bezier_pts: 'list[float]', center_pt: 'list[float]') -> 'list[float]':
        '''
        :param bezier_pts: a list of 16 floats representing 8 control points of the bezier curve: [
                upper_x1, upper_y1,
                upper_x2, upper_y2,
                upper_x3, upper_y3,
                upper_x4, upper_y4,

                lower_x1, lower_y1,
                lower_x2, lower_y2,
                lower_x3, lower_y3,
                lower_x4, lower_y4
            ]
        '''
        centralized_pts = []
        for i in range(len(bezier_pts)):
            if i % 2 == 0:
                centralized_pts.append(bezier_pts[i] - center_pt[0])
            else:
                centralized_pts.append(bezier_pts[i] - center_pt[1])

        return centralized_pts

    def yield_samples(self, img_ids, sample_ratio, augmentation=True, variant_length=False):

        yielded = []

        for img_id in img_ids:

            samples = self.sample_ratio(img_id, sample_ratio)

            for sample in samples:
                query_text = sample['word']['rec']
                query_bezier_centralized = self.bezier_centralized(sample['word']['bezier'], sample['word']['center'])
                query_bezier = sample['word']['bezier']
                query_font = sample['word']['font_embed']
                query_id_in_group = sample['word']['id_in_group']
                neighbour_text = []
                neighbour_bezier_centralized = []
                neighbour_bezier = []
                neighbour_of_the_same_group = []
                neighbour_font = []
                neighbour_id_in_group = []

                for neighbour in sample['dictionary']:
                    neighbour_text.append(neighbour['rec'])
                    neighbour_bezier.append(neighbour['bezier'])
                    neighbour_bezier_centralized.append(
                        self.bezier_centralized(neighbour['bezier'], sample['word']['center']))
                    neighbour_of_the_same_group.append(int(neighbour['group_id'] == sample['word']['group_id']))
                    neighbour_font.append(neighbour['font_embed'])
                    neighbour_id_in_group.append(neighbour['id_in_group'])

                # blend query and neighbour
                source_text = neighbour_text + [query_text]
                source_bezier_centralized = neighbour_bezier_centralized + [query_bezier_centralized]
                source_bezier = neighbour_bezier + [query_bezier]
                source_toponym_mask = neighbour_of_the_same_group + [1]
                source_font = neighbour_font + [query_font]
                source_id_in_group = neighbour_id_in_group + [query_id_in_group]

                if augmentation:
                    if 1 in neighbour_of_the_same_group:
                        n_augmentations = 20
                    else:
                        n_augmentations = 3
                else:
                    n_augmentations = 1


                if variant_length:

                    # extract the ids of toponym and randomly sample some noisy neighbours
                    toponym_ids_ = [i for i, t in enumerate(source_toponym_mask) if t == 1]
                    non_toponym_ids = [i for i, t in enumerate(source_toponym_mask) if t != 1]

                    # sample a random number of non toponym ids
                    num_to_sample = random.randint(0, len(non_toponym_ids))
                    sampled_non_toponym_ids = random.sample(non_toponym_ids, num_to_sample)

                    synthesized_ids_in_list = toponym_ids_ + sampled_non_toponym_ids

                    source_text_ = []
                    source_bezier_centralized_ = []
                    source_bezier_ = []
                    source_toponym_mask_ = []
                    source_font_ = []
                    source_id_in_group_ = []
                    for idx in synthesized_ids_in_list:
                        source_text_.append(source_text[idx])
                        source_bezier_.append(source_bezier[idx])
                        source_bezier_centralized_.append(source_bezier_centralized[idx])
                        source_toponym_mask_.append(source_toponym_mask[idx])
                        source_font_.append(source_font[idx])
                        source_id_in_group_.append(source_id_in_group[idx])
                    query_id_in_sythesized_list = len(toponym_ids_) - 1

                    indices_source = list(range(len(source_text_)))
                    for _ in range(n_augmentations):
                        random.shuffle(indices_source)

                        indices_no_query = [i for i in indices_source if i != query_id_in_sythesized_list]
                        neighbour_text_ = [source_text_[i] for i in indices_no_query]
                        neighbour_bezier_centralized_ = [source_bezier_centralized_[i] for i in indices_no_query]
                        neighbour_bezier_ = [source_bezier_[i] for i in indices_no_query]
                        neighbour_of_the_same_group_ = [source_toponym_mask_[i] for i in indices_no_query]
                        neighbour_font_ = [source_font_[i] for i in indices_no_query]

                        # blend query and neighbour
                        source_text_ = [source_text_[i] for i in indices_source]
                        source_bezier_centralized_ = [source_bezier_centralized_[i] for i in indices_source]
                        source_bezier_ = [source_bezier_[i] for i in indices_source]
                        source_toponym_mask_ = [source_toponym_mask_[i] for i in indices_source]
                        source_font_ = [source_font_[i] for i in indices_source]
                        source_id_in_group_ = [source_id_in_group_[i] for i in indices_source]

                        # get toponym id in reading order
                        toponym_id_in_source_ = [i for i, v in enumerate(source_toponym_mask_) if v == 1]
                        toponym_id_in_group_ = [source_id_in_group_[i] for i in toponym_id_in_source_]
                        toponym_id_sorted_in_source_ = [x[1] for x in
                                                        sorted(list(zip(toponym_id_in_group_, toponym_id_in_source_)),
                                                               key=lambda x: x[0])]
                        query_id_in_source_ = indices_source.index(query_id_in_sythesized_list)
                        toponym_len = len(toponym_id_sorted_in_source_)
                        # will add sot: 0, eot: 1 to the front in batch processing

                        source_len = len(source_text_)

                        yielded.append({'query_text': query_text,
                                        'query_bezier_centralized': query_bezier_centralized,
                                        'neighbour_text': neighbour_text_,
                                        'neighbour_bezier_centralized': neighbour_bezier_centralized_,
                                        'neighbour_of_the_same_group': neighbour_of_the_same_group_,
                                        'query_bezier': query_bezier,
                                        'neighbour_bezier': neighbour_bezier_,
                                        'query_font': query_font,
                                        'neighbour_font': neighbour_font_,
                                        'source_text': source_text_,
                                        'source_bezier_centralized': source_bezier_centralized_,
                                        'source_bezier': source_bezier_,
                                        'source_toponym_mask': source_toponym_mask_,
                                        'source_font': source_font_,
                                        'toponym_id_sorted_in_source': toponym_id_sorted_in_source_,
                                        'query_id_in_source': query_id_in_source_,
                                        'img_id': img_id,
                                        'toponym_len': toponym_len,
                                        'source_len': source_len})

                else:
                    indices_source = list(range(len(source_text)))
                    for _ in range(n_augmentations):
                        random.shuffle(indices_source)

                        indices_no_query = [i for i in indices_source if i != len(source_text)-1]
                        neighbour_text_ = [neighbour_text[i] for i in indices_no_query]
                        neighbour_bezier_centralized_ = [neighbour_bezier_centralized[i] for i in indices_no_query]
                        neighbour_bezier_ = [neighbour_bezier[i] for i in indices_no_query]
                        neighbour_of_the_same_group_ = [neighbour_of_the_same_group[i] for i in indices_no_query]
                        neighbour_font_ = [neighbour_font[i] for i in indices_no_query]
                        # blend query and neighbour
                        source_text_ = [source_text[i] for i in indices_source]
                        source_bezier_centralized_ = [source_bezier_centralized[i] for i in indices_source]
                        source_bezier_ = [source_bezier[i] for i in indices_source]
                        source_toponym_mask_ = [source_toponym_mask[i] for i in indices_source]
                        source_font_ = [source_font[i] for i in indices_source]
                        source_id_in_group_ = [source_id_in_group[i] for i in indices_source]
                        # get toponym id in reading order
                        toponym_id_in_source_ = [i for i, v in enumerate(source_toponym_mask_) if v==1]
                        toponym_id_in_group_ = [source_id_in_group_[i] for i in toponym_id_in_source_]
                        toponym_id_sorted_in_source_ = [x[1] for x in sorted(list(zip(toponym_id_in_group_, toponym_id_in_source_)), key=lambda x: x[0])]
                        query_id_in_source_ = indices_source.index(len(source_text)-1)
                        # add sot, eot, and paddings
                        toponym_len = len(toponym_id_sorted_in_source_)
                        toponym_id_sorted_in_source_ = [len(source_text_)] + toponym_id_sorted_in_source_ + [len(source_text_)+1] + [len(source_text_)+2] * (len(source_text) - len(toponym_id_sorted_in_source_))

                        yielded.append({'query_text': query_text,
                                        'query_bezier_centralized': query_bezier_centralized,
                                        'neighbour_text': neighbour_text_,
                                        'neighbour_bezier_centralized': neighbour_bezier_centralized_,
                                        'neighbour_of_the_same_group': neighbour_of_the_same_group_,
                                        'query_bezier': query_bezier,
                                        'neighbour_bezier': neighbour_bezier_,
                                        'query_font': query_font,
                                        'neighbour_font': neighbour_font_,
                                        'source_text': source_text_,
                                        'source_bezier_centralized': source_bezier_centralized_,
                                        'source_bezier': source_bezier_,
                                        'source_toponym_mask': source_toponym_mask_,
                                        'source_font': source_font_,
                                        'toponym_id_sorted_in_source': toponym_id_sorted_in_source_,
                                        'query_id_in_source': query_id_in_source_,
                                        'img_id': img_id,
                                        'toponym_len': toponym_len})

        return yielded

    def get_train_test_set(self, train_ratio=0.9, sample_ratio=1.0, random_seed=42):

        # filter out images with no group labels
        valid_image_ids = []
        for i in range(len(self.annos)):
            num_groups = len(self.annos[i])
            num_words = 0
            for group in self.annos[i]:
                num_words += len(group['bezier'])
            if num_groups != num_words:
                valid_image_ids.append(i)

        train_img_ids, test_img_ids = train_test_split(valid_image_ids, train_size=train_ratio, test_size=1-train_ratio, random_state=random_seed)
        self.train_set = self.yield_samples(train_img_ids, sample_ratio=1.0, augmentation=True, variant_length=True)
        self.test_set = self.yield_samples(test_img_ids, sample_ratio=sample_ratio, augmentation=False, variant_length=True)

        return self.train_set, self.test_set

    def predict_plot(self,
                    query_pts,
                    query_text,
                    neighbour_pts: list,
                    neighbour_text,
                    gt_label: 'list[int]',
                    predicted_label: 'list[int]',
                    img_id: int,
                    img_name: int,
                    predicted_query_order=-1,
                    predicted_neighbour_order=[-1] * 15):
        '''
        :param neighbour_pts: a list of lists, each list consists of 16 floats representing 8 control points
                of the upper and lower bezier curves of a polygon: [
                upper_x1, upper_y1,
                upper_x2, upper_y2,
                upper_x3, upper_y3,
                upper_x4, upper_y4,

                lower_x1, lower_y1,
                lower_x2, lower_y2,
                lower_x3, lower_y3,
                lower_x4, lower_y4
            ]
                gt_label, predicted_label: a list of the same length as bezier_pts, each element is a binary variable,
                indicating the class of the polygon
        :return a plot of bezier curves, the curves of polygons with gt_label 1 are colored yellow;
        the curves of polygons with predicted_label 1 are colored green; others are colored blue
        '''
        # Assuming self.images, self.image_root_folder, query_pts, query_text, neighbour_pts, gt_label, predicted_label, neighbour_text, img_name are defined
        image = self.images[img_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        background = Image.open(image_path).convert("RGBA")

        # Create a drawing context
        alpha = 0.7  # Set the desired transparency level (0.0 to 1.0)
        enhancer = ImageEnhance.Brightness(background.split()[3])
        background.putalpha(enhancer.enhance(alpha))
        draw = ImageDraw.Draw(background)

        # Plot the query
        upper_half = [(query_pts[i], query_pts[i + 1]) for i in range(0, 8, 2)]
        lower_half = [(query_pts[i], query_pts[i + 1]) for i in range(8, 16, 2)]

        upper_half_x = [p[0] for p in upper_half]
        upper_half_y = [p[1] for p in upper_half]
        lower_half_x = [p[0] for p in lower_half]
        lower_half_y = [p[1] for p in lower_half]

        upper_half_x, upper_half_y = bezier_to_polyline(upper_half_x, upper_half_y)
        lower_half_x, lower_half_y = bezier_to_polyline(lower_half_x, lower_half_y)

        draw.line(list(zip(upper_half_x, upper_half_y)), fill="orange", width=3)
        draw.line(list(zip(lower_half_x, lower_half_y)), fill="orange", width=3)
        draw.text((lower_half[0][0], lower_half[0][1]),
                  query_text + " Order: {}".format(predicted_query_order),
                  fill="orange")

        x_min = min(min(upper_half_x), min(lower_half_x))
        x_max = max(max(upper_half_x), max(lower_half_x))
        y_min = min(min(upper_half_y), min(lower_half_y))
        y_max = max(max(upper_half_y), max(lower_half_y))

        for i, pts in enumerate(neighbour_pts):
            upper_half = [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]
            lower_half = [(pts[i], pts[i + 1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            upper_half_x, upper_half_y = bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = bezier_to_polyline(lower_half_x, lower_half_y)

            if gt_label[i] == 1 and predicted_label[i] == 1:
                color = 'green'
            elif gt_label[i] == 1 and predicted_label[i] == 0:
                color = 'blue'
            elif gt_label[i] == 0 and predicted_label[i] == 1:
                color = 'red'
            else:
                color = 'grey'

            draw.line(list(zip(upper_half_x, upper_half_y)), fill=color, width=3)
            draw.line(list(zip(lower_half_x, lower_half_y)), fill=color, width=3)
            draw.text((lower_half[0][0], lower_half[0][1]),
                      neighbour_text[i] + " Order: {}".format(predicted_neighbour_order[i]),
                      fill=color)

            x_min = min(x_min, min(upper_half_x), min(lower_half_x))
            x_max = max(x_max, max(upper_half_x), max(lower_half_x))
            y_min = min(y_min, min(upper_half_y), min(lower_half_y))
            y_max = max(y_max, max(upper_half_y), max(lower_half_y))

        try:
            cropped_image = background.crop((x_min-10, y_min-10, x_max+10, y_max+10))
        except:
            cropped_image = background.crop((x_min, y_min, x_max, y_max))
        cropped_image.save(f'plots/{img_name}.png')


if __name__ == '__main__':
    import english_encoding as Encoding

    if True:
        grouper = GrouperDataset("train_images")

        grouper.load_annotations_from_file("train_96voc_embed.json")

        grouper.draw_annotations(196, Encoding.decode_text_96)

        cnt = 0
        for i in range(len(grouper.annos)):
            num_groups = len(grouper.annos[i])
            num_words = 0
            for group in grouper.annos[i]:
                num_words += len(group['bezier'])
            if num_groups == num_words:
                print(i)
                print(grouper.images[i]['file_name'])
                print('')
                cnt+=1
        print(cnt)

    if False:
        sorter = SorterDataset("dataset/sorter/train_images")

        sorter.load_annotations_from_file("dataset/sorter/train_96voc_embed.json")

        #sorter.gen_font_embed_all('models/DeepFontEncoder.pth', 'cuda')

        #sorter.save_annotations_to_file("dataset/sorter/train_96voc_embed.json")
        image_id = 3

        samples = sorter.sample(image_id, 2000, closest_pts_count=15, non_overlap=True)

        for i in range(5):
            sorter.draw_sample(image_id, samples[i], Encoding.decode_text_96)

        # Percentage of 'complete' samples
        print(sum([sample['complete'] for sample in samples])/len(samples))