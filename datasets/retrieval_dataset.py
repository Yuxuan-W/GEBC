import argparse
import os
import h5py
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.basic_utils import load_json, load_from_yaml_file, time_sampling
from pytorch_transformers import BertTokenizer


def generate_difference(feat_dict, is_action=False):
    difference_seq = []
    for idx_be in range(feat_dict['before'].shape[0]):
        for idx_af in range(feat_dict['after'].shape[0]):
            difference_seq.append(np.concatenate((feat_dict['before'][idx_be], feat_dict['after'][idx_af])))
    if not is_action:
        for idx_be in range(feat_dict['before'].shape[0]):
            difference_seq.append(np.concatenate((feat_dict['before'][idx_be], feat_dict['boundary'][0])))
        for idx_af in range(feat_dict['after'].shape[0]):
            difference_seq.append(np.concatenate((feat_dict['boundary'][0], feat_dict['after'][idx_af])))
    return np.array(difference_seq)


class RetrievalDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        assert split in ['train', 'test', 'val'], "Invalid split: split must in 'train', 'test' and 'val'."
        self.yaml_file = args.yaml_file
        self.cfg = load_from_yaml_file(self.yaml_file)
        self.is_train = split == 'train'
        self.triplet = False

        if split == 'train':
            self.annotation = load_json(self.cfg['train_annotation'])
        elif split == 'test':
            self.annotation = load_json(self.cfg['test_annotation'])
        else:
            self.annotation = load_json(self.cfg['val_annotation'])

        if split == 'train' and self.triplet:
            self.boundary_list = self.build_boundary_triplet_list()
        else:
            self.boundary_list = self.build_boundary_list()
        self.region_feature_root = self.cfg['region_feature_root']
        self.action_feature_root = self.cfg['action_feature_root']

        self.max_token_length = args.max_token_length
        self.tokenizer = tokenizer

        self.max_frame_num = args.max_frame_num
        self.max_object_per_frame = args.max_object_per_frame
        self.max_object_length = (2 * self.max_frame_num + 1) * self.max_object_per_frame
        self.max_frame_length = 2 * self.max_frame_num + 1
        self.max_frame_difference_length = self.max_frame_num ** 2 + self.max_frame_num * 2
        self.max_action_length = args.max_action_length
        self.max_seq_length = self.max_token_length + self.max_object_length + \
                              self.max_frame_length + self.max_frame_difference_length + self.max_action_length

    def __len__(self):
        return len(self.boundary_list)

    def __getitem__(self, idx):
        boundary = self.boundary_list[idx]
        boundary_id = boundary['boundary_id']
        video_id = boundary_id[:11]
        boundary_idx = int(boundary_id.split('_')[-1][:-1])
        timestamps = [boundary['prev_timestamp'], boundary['timestamp'], boundary['next_timestamp']]
        caption = boundary['caption']

        if self.is_train and self.triplet:
            # positive
            frame_scale, frame_feat, obj_bbox, obj_feat = self.get_regions(video_id, timestamps)
            act_feat = self.get_actions(video_id, boundary_idx)
            pos_examples = self.tensorize_example(caption, obj_feat, obj_bbox, frame_feat, act_feat)

            video_id = boundary['intra_neg']['boundary_id'][:11]
            boundary_idx = int(boundary['intra_neg']['boundary_id'].split('_')[-1])
            timestamps = [boundary['intra_neg']['prev_timestamp'], boundary['intra_neg']['timestamp'],
                          boundary['intra_neg']['next_timestamp']]
            frame_scale, frame_feat, obj_bbox, obj_feat = self.get_regions(video_id, timestamps)
            act_feat = self.get_actions(video_id, boundary_idx)
            intra_neg_examples = self.tensorize_example(caption, obj_feat, obj_bbox, frame_feat, act_feat)

            video_id = boundary['inter_neg']['boundary_id'][:11]
            boundary_idx = int(boundary['inter_neg']['boundary_id'].split('_')[-1])
            timestamps = [boundary['inter_neg']['prev_timestamp'], boundary['inter_neg']['timestamp'],
                          boundary['inter_neg']['next_timestamp']]
            frame_scale, frame_feat, obj_bbox, obj_feat = self.get_regions(video_id, timestamps)
            act_feat = self.get_actions(video_id, boundary_idx)
            inter_neg_examples = self.tensorize_example(caption, obj_feat, obj_bbox, frame_feat, act_feat)

            examples = self.package_triplet(pos_examples, intra_neg_examples, inter_neg_examples)
        else:
            frame_scale, frame_feat, obj_bbox, obj_feat = self.get_regions(video_id, timestamps)
            act_feat = self.get_actions(video_id, boundary_idx)
            examples = self.tensorize_example(caption, obj_feat, obj_bbox, frame_feat, act_feat)

        return boundary_id, examples

    def get_caption(self):
        caption_dict = dict()
        for boundary in self.boundary_list:
            caption_dict[boundary['boundary_id']] = [boundary['caption']]
        return caption_dict

    def build_boundary_triplet_list(self):
        boundary_list = []
        for vid, boundaries in self.annotation.items():
            for b_idx in range(len(boundaries)):
                boundary = boundaries[b_idx]

                if len(boundaries) > 1:
                    neg_b_idx = b_idx
                    while neg_b_idx == b_idx:
                        neg_b_idx = random.randint(0, len(boundaries) - 1)
                    intra_neg = boundaries[neg_b_idx]
                else:
                    inter_neg_vid_alter = vid
                    while vid == inter_neg_vid_alter:
                        inter_neg_vid_alter = random.sample(list(self.annotation.keys()), 1)[0]
                        inter_neg_b_idx_alter = random.randint(0, len(self.annotation[inter_neg_vid_alter]) - 1)
                    intra_neg = self.annotation[inter_neg_vid_alter][inter_neg_b_idx_alter]

                inter_neg_vid = vid
                while vid == inter_neg_vid:
                    inter_neg_vid = random.sample(list(self.annotation.keys()), 1)[0]
                    inter_neg_b_idx = random.randint(0, len(self.annotation[inter_neg_vid]) - 1)
                inter_neg = self.annotation[inter_neg_vid][inter_neg_b_idx]

                boundary_list.append({
                    **boundary,
                    'intra_neg': intra_neg,
                    'inter_neg': inter_neg
                })

        return boundary_list

    def build_boundary_list(self):
        boundary_list = []
        for vid, boundaries in self.annotation.items():
            for b in boundaries:
                boundary_list.append(b)
        return boundary_list

    def package_triplet(self, pos_examples, intra_neg_examples, inter_neg_examples):
        input_ids = torch.stack((pos_examples[0], intra_neg_examples[0], inter_neg_examples[0]), dim=0)
        attention_mask = torch.stack((pos_examples[1], intra_neg_examples[1], inter_neg_examples[1]), dim=0)
        encoded_obj_feat = torch.stack((pos_examples[2], intra_neg_examples[2], inter_neg_examples[2]), dim=0)
        encoded_frame_feat = torch.stack((pos_examples[3], intra_neg_examples[3], inter_neg_examples[3]), dim=0)
        encoded_frame_feat_diff = torch.stack((pos_examples[4], intra_neg_examples[4], inter_neg_examples[4]), dim=0)
        encoded_act_feat = torch.stack((pos_examples[5], intra_neg_examples[5], inter_neg_examples[5]), dim=0)
        encoded_act_feat_diff = torch.stack((pos_examples[6], intra_neg_examples[6], inter_neg_examples[6]), dim=0)
        return (input_ids, attention_mask, encoded_obj_feat, encoded_frame_feat,
                encoded_frame_feat_diff, encoded_act_feat, encoded_act_feat_diff)

    def get_regions(self, vid, timestamps):
        filename = os.path.join(self.region_feature_root, vid + '.hdf5')
        with h5py.File(filename, 'r') as h5:
            time_list = h5.keys()
            sampled_time = time_sampling(self.max_frame_num, time_list, timestamps)
            frame_feat = dict(
                boundary=[h5[f"{sampled_time['boundary']:.3f}"]['feature'][0]],
                before=[h5[f'{time:.3f}']['feature'][0] for time in sampled_time['before']],
                after=[h5[f'{time:.3f}']['feature'][0] for time in sampled_time['after']]
            )
            object_feat = dict(
                boundary=[h5[f"{sampled_time['boundary']:.3f}"]['feature'][1:]],
                before=[h5[f'{time:.3f}']['feature'][1:] for time in sampled_time['before']],
                after=[h5[f'{time:.3f}']['feature'][1:] for time in sampled_time['after']]
            )
            frame_scale = h5[f"{sampled_time['boundary']:.3f}"]['bbox'][0, 2:4]
            scale_factor = np.array([frame_scale[0], frame_scale[1], frame_scale[0], frame_scale[1], 1])
            object_bbox = dict(
                boundary=[h5[f"{sampled_time['boundary']:.3f}"]['bbox'][1:] / scale_factor],
                before=[h5[f'{time:.3f}']['bbox'][1:] / scale_factor for time in sampled_time['before']],
                after=[h5[f'{time:.3f}']['bbox'][1:] / scale_factor for time in sampled_time['after']]
            )
        return frame_scale, frame_feat, object_bbox, object_feat

    def get_actions(self, vid, boundary_idx):
        filename = os.path.join(self.action_feature_root, vid + '.hdf5')
        with h5py.File(filename, 'r') as h5:
            action_feat = dict(
                before=h5[str(boundary_idx)][:],
                after=h5[str(boundary_idx + 1)][:]
            )
        return action_feat

    def tensorize_example(self, caption, obj_feat, obj_bbox, frame_feat, act_feat):
        # encode caption
        input_ids, tokens_len = self.caption_encoding(caption)

        # encode object feature
        encoded_obj_feat = dict(
            before=self.video_feature_encoding('before', obj_feat['before'], obj_bbox['before']),
            boundary=self.video_feature_encoding('boundary', obj_feat['boundary'], obj_bbox['boundary']),
            after=self.video_feature_encoding('after', obj_feat['after'], obj_bbox['after'])
        )

        # encode frame feature
        encoded_frame_feat = dict(
            before=self.video_feature_encoding('before', frame_feat['before']),
            boundary=self.video_feature_encoding('boundary', frame_feat['boundary']),
            after=self.video_feature_encoding('after', frame_feat['after'])
        )

        # encode action feature
        encoded_act_feat = dict(
            before=self.video_feature_encoding('before', act_feat['before'], is_action=True),
            after=self.video_feature_encoding('after', act_feat['after'], is_action=True)
        )

        # feature difference
        encoded_frame_feat_diff = generate_difference(encoded_frame_feat)
        encoded_act_feat_diff = generate_difference(encoded_act_feat, is_action=True)

        # zero padding
        encoded_obj_feat, obj_len = self.zero_padding(encoded_obj_feat, feat_type='obj')
        encoded_frame_feat, frame_len = self.zero_padding(encoded_frame_feat, feat_type='frame')
        encoded_frame_feat_diff, frame_diff_len = self.zero_padding(encoded_frame_feat_diff, feat_type='frame_diff')
        encoded_act_feat = np.concatenate((encoded_act_feat['before'], encoded_act_feat['after']), axis=0)

        # attention mask
        attention_mask = self.generate_attention_mask(tokens_len, obj_len, frame_len, frame_diff_len)

        # tensorize
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        encoded_obj_feat = torch.tensor(encoded_obj_feat, dtype=torch.float32)
        encoded_frame_feat = torch.tensor(encoded_frame_feat, dtype=torch.float32)
        encoded_frame_feat_diff = torch.tensor(encoded_frame_feat_diff, dtype=torch.float32)
        encoded_act_feat = torch.tensor(encoded_act_feat, dtype=torch.float32)
        encoded_act_feat_diff = torch.tensor(encoded_act_feat_diff, dtype=torch.float32)

        return input_ids, attention_mask, encoded_obj_feat, encoded_frame_feat, \
               encoded_frame_feat_diff, encoded_act_feat, encoded_act_feat_diff

    def caption2tokens(self, caption, splitter='//'):
        tokens = [self.tokenizer.cls_token]
        splitted_caption = caption.split(splitter)
        assert len(splitted_caption) == 3, "Invalid: Caption has more than 3 parts: " + caption
        tokens += self.tokenizer.tokenize(caption) + [self.tokenizer.sep_token]
        assert len(tokens) <= self.max_token_length, "Too long caption:" + caption
        return tokens

    def caption_encoding(self, caption):
        tokens = self.caption2tokens(caption)
        tokens_len = len(tokens)

        # pad on the right for image captioning, which means pad on the right of [[caption], [pad]]
        padding_len = self.max_token_length - tokens_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return input_ids, tokens_len

    def video_feature_encoding(self, segment, feat, bbox=None, is_action=False):
        assert segment in ['before', 'boundary', 'after'], "Invalid: Segment must in 'before', 'boundary' and 'after'."
        encoded_feat = []
        if is_action:
            encoded_feat.append(np.expand_dims(feat, axis=0))
        else:
            for frame_idx in range(len(feat)):
                frame_feat = np.expand_dims(feat[frame_idx], axis=0) if bbox is None else feat[frame_idx]
                frame_bbox = None if bbox is None else bbox[frame_idx]
                if segment == 'before':
                    temporal_dist = (1 - frame_idx / len(feat)) * np.ones((frame_feat.shape[0], 1))
                elif segment == 'after':
                    temporal_dist = (frame_idx + 1) / len(feat) * np.ones((frame_feat.shape[0], 1))
                else:
                    temporal_dist = np.zeros((frame_feat.shape[0], 1))

                if bbox is None:
                    encoded_feat.append(np.concatenate((frame_feat, temporal_dist), axis=1))
                else:
                    encoded_feat.append(np.concatenate((frame_feat[:self.max_object_per_frame],
                                                        frame_bbox[:self.max_object_per_frame],
                                                        temporal_dist[:self.max_object_per_frame]), axis=1)
                                        if frame_feat.shape[0] > self.max_object_per_frame else
                                        np.concatenate((frame_feat, frame_bbox, temporal_dist), axis=1))
        encoded_feat = np.concatenate(encoded_feat, axis=0)
        if segment == 'before':
            segment_encoding = 0 * np.ones((encoded_feat.shape[0], 1))
        elif segment == 'after':
            segment_encoding = 2 * np.ones((encoded_feat.shape[0], 1))
        else:
            segment_encoding = 1 * np.ones((encoded_feat.shape[0], 1))
        encoded_feat = np.concatenate((encoded_feat, segment_encoding), axis=1)
        return encoded_feat

    def zero_padding(self, feat_dict, feat_type):
        assert feat_type in ['obj', 'frame', 'frame_diff']

        if feat_type in ['obj', 'frame']:
            padded_seq = []
            for _, feat in feat_dict.items():
                padded_seq.append(feat)
            padded_seq = np.concatenate(padded_seq, axis=0)
            feat_len = padded_seq.shape[0]
            if feat_type == 'obj':
                max_length = self.max_object_length
                if feat_len < max_length:
                    padded_seq = np.concatenate((padded_seq, np.zeros((max_length - feat_len, padded_seq.shape[1]))))
            else:
                max_length = self.max_frame_length
                if padded_seq.shape[0] < max_length:
                    padded_seq = np.concatenate((padded_seq, np.zeros((max_length - feat_len, padded_seq.shape[1]))))
        else:
            padded_seq = feat_dict
            feat_len = padded_seq.shape[0]
            max_length = self.max_frame_difference_length
            if padded_seq.shape[0] < max_length:
                padded_seq = np.concatenate(
                    (padded_seq, np.zeros((max_length - padded_seq.shape[0], padded_seq.shape[1]))))
        return padded_seq, feat_len

    def generate_attention_mask(self, tokens_len, obj_len, frame_len, frame_diff_len):
        c_start, c_end = 0, tokens_len
        obj_start = self.max_token_length
        obj_end = self.max_token_length + obj_len
        frame_start = obj_start + self.max_object_length
        frame_end = obj_start + self.max_object_length + frame_len
        frame_diff_start = frame_start + self.max_frame_length
        frame_diff_end = frame_start + self.max_frame_length + frame_diff_len
        act_start = frame_diff_start + self.max_frame_difference_length
        act_end = frame_diff_start + self.max_frame_difference_length + self.max_action_length
        assert act_end == self.max_seq_length

        attention_mask = np.zeros((self.max_seq_length, self.max_seq_length))

        # full attention for all, will separate and only keep 2 diagonal parts in the model
        range_0_list = [[c_start, c_end], [obj_start, obj_end], [frame_start, frame_end], [frame_diff_start, frame_diff_end], [act_start, act_end]]
        range_1_list = [[c_start, c_end], [obj_start, obj_end], [frame_start, frame_end], [frame_diff_start, frame_diff_end], [act_start, act_end]]
        for range_0 in range_0_list:
            for range_1 in range_1_list:
                attention_mask[range_0[0]:range_0[1], range_1[0]:range_1[1]] = 1
        return attention_mask


class RetrievalCorpusDataset(Dataset):
    def __init__(self, args, split):
        assert split in ['train', 'test', 'val'], "Invalid split: split must in 'train', 'test' and 'val'."
        self.yaml_file = args.yaml_file
        self.cfg = load_from_yaml_file(self.yaml_file)
        self.is_train = split == 'train'

        if split == 'train':
            self.annotation = load_json(self.cfg['train_gebd_proposals'])
        elif split == 'test':
            self.annotation = load_json(self.cfg['test_gebd_proposals'])
        else:
            self.annotation = load_json(self.cfg['val_gebd_proposals'])

        self.boundary_list = self.build_boundary_list()
        self.region_feature_root = self.cfg['region_feature_root']
        self.action_feature_root = self.cfg['action_gebd_feature_root']

        self.max_frame_num = args.max_frame_num
        self.max_object_per_frame = args.max_object_per_frame
        self.max_object_length = (2 * self.max_frame_num + 1) * self.max_object_per_frame
        self.max_frame_length = 2 * self.max_frame_num + 1
        self.max_frame_difference_length = self.max_frame_num ** 2 + self.max_frame_num * 2
        self.max_action_length = args.max_action_length
        self.max_seq_length = self.max_object_length + self.max_frame_length + \
                              self.max_frame_difference_length + self.max_action_length

    def __len__(self):
        return len(self.boundary_list)

    def build_boundary_list(self):
        boundary_list = []
        for vid, boundaries in self.annotation.items():
            for b in boundaries:
                boundary_list.append(b)
        return boundary_list

    def __getitem__(self, idx):
        boundary = self.boundary_list[idx]
        boundary_id = boundary['boundary_id']
        video_id = boundary_id[:11]
        boundary_idx = int(boundary_id.split('_')[-1])
        timestamps = [boundary['prev_timestamp'], boundary['timestamp'], boundary['next_timestamp']]

        frame_scale, frame_feat, obj_bbox, obj_feat = self.get_regions(video_id, timestamps)
        act_feat = self.get_actions(video_id, boundary_idx)
        examples = self.tensorize_example(obj_feat, obj_bbox, frame_feat, act_feat)

        return boundary_id, examples

    def get_regions(self, vid, timestamps):
        filename = os.path.join(self.region_feature_root, vid + '.hdf5')
        with h5py.File(filename, 'r') as h5:
            time_list = h5.keys()
            sampled_time = time_sampling(self.max_frame_num, time_list, timestamps)
            frame_feat = dict(
                boundary=[h5[f"{sampled_time['boundary']:.3f}"]['feature'][0]],
                before=[h5[f'{time:.3f}']['feature'][0] for time in sampled_time['before']],
                after=[h5[f'{time:.3f}']['feature'][0] for time in sampled_time['after']]
            )
            object_feat = dict(
                boundary=[h5[f"{sampled_time['boundary']:.3f}"]['feature'][1:]],
                before=[h5[f'{time:.3f}']['feature'][1:] for time in sampled_time['before']],
                after=[h5[f'{time:.3f}']['feature'][1:] for time in sampled_time['after']]
            )
            frame_scale = h5[f"{sampled_time['boundary']:.3f}"]['bbox'][0, 2:4]
            scale_factor = np.array([frame_scale[0], frame_scale[1], frame_scale[0], frame_scale[1], 1])
            object_bbox = dict(
                boundary=[h5[f"{sampled_time['boundary']:.3f}"]['bbox'][1:] / scale_factor],
                before=[h5[f'{time:.3f}']['bbox'][1:] / scale_factor for time in sampled_time['before']],
                after=[h5[f'{time:.3f}']['bbox'][1:] / scale_factor for time in sampled_time['after']]
            )
        return frame_scale, frame_feat, object_bbox, object_feat

    def get_actions(self, vid, boundary_idx):
        filename = os.path.join(self.action_feature_root, vid + '.hdf5')
        with h5py.File(filename, 'r') as h5:
            action_feat = dict(
                before=h5[str(boundary_idx)][:],
                after=h5[str(boundary_idx + 1)][:]
            )
        return action_feat

    def tensorize_example(self, obj_feat, obj_bbox, frame_feat, act_feat):

        # encode object feature
        encoded_obj_feat = dict(
            before=self.video_feature_encoding('before', obj_feat['before'], obj_bbox['before']),
            boundary=self.video_feature_encoding('boundary', obj_feat['boundary'], obj_bbox['boundary']),
            after=self.video_feature_encoding('after', obj_feat['after'], obj_bbox['after'])
        )

        # encode frame feature
        encoded_frame_feat = dict(
            before=self.video_feature_encoding('before', frame_feat['before']),
            boundary=self.video_feature_encoding('boundary', frame_feat['boundary']),
            after=self.video_feature_encoding('after', frame_feat['after'])
        )

        # encode action feature
        encoded_act_feat = dict(
            before=self.video_feature_encoding('before', act_feat['before'], is_action=True),
            after=self.video_feature_encoding('after', act_feat['after'], is_action=True)
        )

        # feature difference
        encoded_frame_feat_diff = generate_difference(encoded_frame_feat)
        encoded_act_feat_diff = generate_difference(encoded_act_feat, is_action=True)

        # zero padding
        encoded_obj_feat, obj_len = self.zero_padding(encoded_obj_feat, feat_type='obj')
        encoded_frame_feat, frame_len = self.zero_padding(encoded_frame_feat, feat_type='frame')
        encoded_frame_feat_diff, frame_diff_len = self.zero_padding(encoded_frame_feat_diff, feat_type='frame_diff')
        encoded_act_feat = np.concatenate((encoded_act_feat['before'], encoded_act_feat['after']), axis=0)

        # attention mask
        attention_mask = self.generate_attention_mask(obj_len, frame_len, frame_diff_len)

        # tensorize
        attention_mask = torch.tensor(attention_mask)
        encoded_obj_feat = torch.tensor(encoded_obj_feat, dtype=torch.float32)
        encoded_frame_feat = torch.tensor(encoded_frame_feat, dtype=torch.float32)
        encoded_frame_feat_diff = torch.tensor(encoded_frame_feat_diff, dtype=torch.float32)
        encoded_act_feat = torch.tensor(encoded_act_feat, dtype=torch.float32)
        encoded_act_feat_diff = torch.tensor(encoded_act_feat_diff, dtype=torch.float32)

        return attention_mask, encoded_obj_feat, encoded_frame_feat, \
               encoded_frame_feat_diff, encoded_act_feat, encoded_act_feat_diff

    def generate_attention_mask(self, obj_len, frame_len, frame_diff_len):

        obj_start = 0
        obj_end = obj_len
        frame_start = obj_start + self.max_object_length
        frame_end = obj_start + self.max_object_length + frame_len
        frame_diff_start = frame_start + self.max_frame_length
        frame_diff_end = frame_start + self.max_frame_length + frame_diff_len
        act_start = frame_diff_start + self.max_frame_difference_length
        act_end = frame_diff_start + self.max_frame_difference_length + self.max_action_length
        assert act_end == self.max_seq_length

        attention_mask = np.zeros((self.max_seq_length, self.max_seq_length))

        # full attention for all, will separate and only keep 2 diagonal parts in the model
        range_0_list = [[obj_start, obj_end], [frame_start, frame_end], [frame_diff_start, frame_diff_end], [act_start, act_end]]
        range_1_list = [[obj_start, obj_end], [frame_start, frame_end], [frame_diff_start, frame_diff_end], [act_start, act_end]]
        for range_0 in range_0_list:
            for range_1 in range_1_list:
                attention_mask[range_0[0]:range_0[1], range_1[0]:range_1[1]] = 1
        return attention_mask

    def video_feature_encoding(self, segment, feat, bbox=None, is_action=False):
        assert segment in ['before', 'boundary', 'after'], "Invalid: Segment must in 'before', 'boundary' and 'after'."
        encoded_feat = []
        if is_action:
            encoded_feat.append(np.expand_dims(feat, axis=0))
        else:
            for frame_idx in range(len(feat)):
                frame_feat = np.expand_dims(feat[frame_idx], axis=0) if bbox is None else feat[frame_idx]
                frame_bbox = None if bbox is None else bbox[frame_idx]
                if segment == 'before':
                    temporal_dist = (1 - frame_idx / len(feat)) * np.ones((frame_feat.shape[0], 1))
                elif segment == 'after':
                    temporal_dist = (frame_idx + 1) / len(feat) * np.ones((frame_feat.shape[0], 1))
                else:
                    temporal_dist = np.zeros((frame_feat.shape[0], 1))

                if bbox is None:
                    encoded_feat.append(np.concatenate((frame_feat, temporal_dist), axis=1))
                else:
                    encoded_feat.append(np.concatenate((frame_feat[:self.max_object_per_frame],
                                                        frame_bbox[:self.max_object_per_frame],
                                                        temporal_dist[:self.max_object_per_frame]), axis=1)
                                        if frame_feat.shape[0] > self.max_object_per_frame else
                                        np.concatenate((frame_feat, frame_bbox, temporal_dist), axis=1))
        encoded_feat = np.concatenate(encoded_feat, axis=0)
        if segment == 'before':
            segment_encoding = 0 * np.ones((encoded_feat.shape[0], 1))
        elif segment == 'after':
            segment_encoding = 2 * np.ones((encoded_feat.shape[0], 1))
        else:
            segment_encoding = 1 * np.ones((encoded_feat.shape[0], 1))
        encoded_feat = np.concatenate((encoded_feat, segment_encoding), axis=1)
        return encoded_feat

    def zero_padding(self, feat_dict, feat_type):
        assert feat_type in ['obj', 'frame', 'frame_diff']

        if feat_type in ['obj', 'frame']:
            padded_seq = []
            for _, feat in feat_dict.items():
                padded_seq.append(feat)
            padded_seq = np.concatenate(padded_seq, axis=0)
            feat_len = padded_seq.shape[0]
            if feat_type == 'obj':
                max_length = self.max_object_length
                if feat_len < max_length:
                    padded_seq = np.concatenate((padded_seq, np.zeros((max_length - feat_len, padded_seq.shape[1]))))
            else:
                max_length = self.max_frame_length
                if padded_seq.shape[0] < max_length:
                    padded_seq = np.concatenate((padded_seq, np.zeros((max_length - feat_len, padded_seq.shape[1]))))
        else:
            padded_seq = feat_dict
            feat_len = padded_seq.shape[0]
            max_length = self.max_frame_difference_length
            if padded_seq.shape[0] < max_length:
                padded_seq = np.concatenate(
                    (padded_seq, np.zeros((max_length - padded_seq.shape[0], padded_seq.shape[1]))))
        return padded_seq, feat_len


if __name__ == '__main__':

    # unit test
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", default='../config/retrieval_config.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--max_token_length", default=90, type=int,
                        help="The max length of caption tokens.")
    parser.add_argument("--max_frame_num", default=10, type=int,
                        help="The max number of frame before or after boundary.")
    parser.add_argument("--max_object_per_frame", default=20, type=int,
                        help="The max object number in single frame.")
    parser.add_argument("--max_action_length", default=3, type=int,
                        help="The max length of action feature, including difference feature.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Probability to mask input sentence during training.")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # dataset = RetrievalDataset(args, tokenizer, 'train')
    dataset = RetrievalCorpusDataset(args, 'test')
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    for batch in tqdm(dataloader):
        a = 1

    dataset = RetrievalCorpusDataset(args, 'val')
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    for batch in tqdm(dataloader):
        a = 1
