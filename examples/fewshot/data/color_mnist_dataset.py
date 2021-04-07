import os
import os.path as osp
import random
import string
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image
import nltk
from nltk.corpus import wordnet
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import collections

class ColorMNIST(Dataset):
    def __init__(self, datapath, split, subsample_num=1):
        np.random.seed(42)
        self.split = split
        self.subsample_num = subsample_num
        fname = "color_mnist_" + self.split + ".pkl"
        color_mnist_datapath = Path(datapath) / fname

        # creating artifical dataset:
        if not color_mnist_datapath.is_file():
            print("started creating new dataset")
            self.mnist = torchvision.datasets.MNIST(datapath, train=self.split == 'train', download=True)

            # we will read all mnist first to have a deterministic blocking of leaf nodes
            t_start = datetime.now()
            mnist_samples = [[img, target] for img, target in self.mnist]
            self.enhanced_mnist_samples = self.do_color_enhancement(mnist_samples)
            pickle.dump(self.enhanced_mnist_samples, open(color_mnist_datapath, "wb"))
            print("done sampling, took", datetime.now() - t_start)
        else:
            self.enhanced_mnist_samples = pickle.load(open(color_mnist_datapath, "rb"))



        # mapping from tuple -> index
        self.hierarchy2index = {}
        i = 0
        for chan in range(0, 3): # different channels
            self.hierarchy2index[(chan,)] = i
            i += 1
            for clazz in range(0, 4): # different shades
                self.hierarchy2index[(chan, clazz)] = i
                i += 1
                for t in range(0, 10): # different digits
                    self.hierarchy2index[(chan, clazz, t)] = i
                    i += 1
        self.num_embeddings = i
        self.index2hierarchy = {v: k for k, v in self.hierarchy2index.items()}
        
        self.positive_edge_indices = []
        self.positive_edge_indices_minimum = []
        for chan in range(0, 3):
            for clazz in range(0, 4):
                self.positive_edge_indices.append((self.hierarchy2index[(chan,)],
                                                   self.hierarchy2index[(chan, clazz)]))
                self.positive_edge_indices_minimum.append((self.hierarchy2index[(chan,)],
                                                   self.hierarchy2index[(chan, clazz)]))
                for t in range(0, 10):
                        self.positive_edge_indices.append((self.hierarchy2index[(chan, clazz)],
                                                           self.hierarchy2index[(chan, clazz, t)]))
                        self.positive_edge_indices_minimum.append((self.hierarchy2index[(chan, clazz)],
                                                           self.hierarchy2index[(chan, clazz, t)]))
                        # full transitive closure
                        self.positive_edge_indices.append((self.hierarchy2index[(chan,)],
                                                           self.hierarchy2index[(chan, clazz, t)]))

        # enumerate all leaf embeddings
        self.class_embeddings = []
        counter = 0
        for chan in range(0,3):
            for clazz in range(0,4):
                for t in range(0, 10):
                    self.class_embeddings.append(self.hierarchy2index[(chan, clazz, t)])


        # sample 5 (u', v) and 5 (u, v') for each positive (u,v) where (parent, child)
        self.negative_children_indices = collections.defaultdict(list)
        for pei in self.positive_edge_indices:
            from_index = pei[0]
            from_hierarchy = self.index2hierarchy[from_index]
            if len(from_hierarchy) == 1:
                # highest layer
                for chan in range(0, 3):
                    for clazz in range(0, 4):
                        if chan != from_hierarchy[0]:
                            self.negative_children_indices[from_index].append(self.hierarchy2index[(chan, clazz)])
                            # full transitive closure
                            for t in range(0, 10):
                                self.negative_children_indices[from_index].append(self.hierarchy2index[(chan, clazz, t)])
            elif len(from_hierarchy) == 2:
                # second layer
                for chan in range(0, 3):
                    for clazz in range(0, 4):
                        for t in range(0, 10):
                            # if chan != from_hierarchy[0] and clazz != from_hierarchy[1]:
                            if not (chan == from_hierarchy[0] and clazz == from_hierarchy[1]):
                                self.negative_children_indices[from_index].append(self.hierarchy2index[(chan, clazz, t)])

            else:
                # last layer of hierarchy, happy child free
                pass


        self.negative_parents_indices = collections.defaultdict(list)
        for pei in self.positive_edge_indices:
            to_index = pei[1]
            to_hierarchy = self.index2hierarchy[to_index]
            if len(to_hierarchy) == 1:
                # highest layer no parents. Should also never be the case.
                pass
            elif len(to_hierarchy) == 2:
                # second layer, add the two other parents
                for chan in range(0, 3):
                    if chan != to_hierarchy[0]:
                        self.negative_parents_indices[to_index].append(self.hierarchy2index[(chan,)])
            else:
                # last layer of hierarchy, add two layers of parents
                for chan in range(0,3):
                    if chan != to_hierarchy[0]:
                        self.negative_parents_indices[to_index].append(self.hierarchy2index[(chan,)])
                    for clazz in range(0, 4):
                        if clazz != to_hierarchy[1]:
                            self.negative_parents_indices[to_index].append(self.hierarchy2index[(chan, clazz)])
                        # for t in range(0, 10):
                        #     if t != to_hierarchy[2]:
                        #         self.negative_parents_indices[to_index].append(self.hierarchy2index[(chan, clazz, t)])

        # create corrupted (x', y) sorted by level
        self.negative_parents_indices_per_layer = collections.defaultdict(list)
        self.negative_parents_indices_layer2_same_layer_same_parent = collections.defaultdict(list)
        # for pei in self.positive_edge_indices:
        #     to_index = pei[1]
        #     to_hierarchy = self.index2hierarchy[to_index]
        #     self.negative_parents_indices_per_layer[to_index] = [[], [], []] # for three levels
        #     if len(to_hierarchy) == 1:
        #         for chan in range(0, 3):
        #             if to_hierarchy[0] != chan:
        #                 self.negative_parents_indices_per_layer[to_index][0].append(self.hierarchy2index[(chan,)])
        #                 for clazz in range(0, 4):
        #                     self.negative_parents_indices_per_layer[to_index][1].append(self.hierarchy2index[(chan,clazz)])
        #                     for t in range(0, 10):
        #                         self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])
        #     elif len(to_hierarchy) == 2:
        #         for chan in range(0, 3):
        #             if to_hierarchy[0] != chan:
        #                 self.negative_parents_indices_per_layer[to_index][0].append(self.hierarchy2index[(chan,)])
        #             for clazz in range(0, 4):
        #                 if not (to_hierarchy[0] == chan and to_hierarchy[1] == clazz):
        #                     self.negative_parents_indices_per_layer[to_index][1].append(self.hierarchy2index[(chan,clazz)])
        #                     for t in range(0, 10):
        #                         self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])
        #     else: # last layer
        #         for chan in range(0, 3):
        #             if to_hierarchy[0] != chan:
        #                 self.negative_parents_indices_per_layer[to_index][0].append(self.hierarchy2index[(chan,)])
        #             for clazz in range(0, 4):
        #                 if not(to_hierarchy[0] == chan and to_hierarchy[1] == clazz):
        #                     self.negative_parents_indices_per_layer[to_index][1].append(self.hierarchy2index[(chan,clazz)])
        #                 for t in range(0, 10):
        #                     if not (to_hierarchy[0] == chan and to_hierarchy[1] == clazz and to_hierarchy[2] == t):
        #                         self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])

        # add lvl 0 as target
        zero = self.hierarchy2index[(0,)]
        one = self.hierarchy2index[(1,)]
        two = self.hierarchy2index[(2,)]
        for pei in self.positive_edge_indices + [(zero, zero), (one, one), (two, two)]:
            to_index = pei[1]
            to_hierarchy = self.index2hierarchy[to_index]
            self.negative_parents_indices_per_layer[to_index] = [[], [], []] # for three levels
            self.negative_parents_indices_layer2_same_layer_same_parent[to_index] = [[], [], []]
            if len(to_hierarchy) == 1:
                for chan in range(0, 3):
                    if to_hierarchy[0] != chan:
                        self.negative_parents_indices_per_layer[to_index][0].append(self.hierarchy2index[(chan,)])
                    for clazz in range(0, 4):
                        self.negative_parents_indices_per_layer[to_index][1].append(self.hierarchy2index[(chan, clazz)])
                        for t in range(0, 10):
                            self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])
            elif len(to_hierarchy) == 2:
                for chan in range(0, 3):
                    if to_hierarchy[0] != chan:
                        self.negative_parents_indices_per_layer[to_index][0].append(self.hierarchy2index[(chan,)])
                    for clazz in range(0, 4):
                        if not (to_hierarchy[0] == chan and to_hierarchy[1] == clazz):
                            self.negative_parents_indices_per_layer[to_index][1].append(self.hierarchy2index[(chan, clazz)])
                        for t in range(0, 10):
                            self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])
            else: # last layer
                for chan in range(0, 3):
                    if to_hierarchy[0] != chan:
                        self.negative_parents_indices_per_layer[to_index][0].append(self.hierarchy2index[(chan,)])
                    for clazz in range(0, 4):
                        if not(to_hierarchy[0] == chan and to_hierarchy[1] == clazz):
                            self.negative_parents_indices_per_layer[to_index][1].append(self.hierarchy2index[(chan,clazz)])
                        for t in range(0, 10):
                            # if not (to_hierarchy[0] == chan and to_hierarchy[1] == clazz and to_hierarchy[2] == t):
                            #     self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])
                            if to_hierarchy[0] == chan and to_hierarchy[1] == clazz and not to_hierarchy[2] == t: # only negative samples from the same parent
                                self.negative_parents_indices_per_layer[to_index][2].append(self.hierarchy2index[(chan, clazz, t)])
        # self.negative_edge_indices_map = {}
        # # from full transitive closure (full outgoing edges, not in opposite direction)
        # for chan0 in range(0, 3):
        #     pos_from = self.hierarchy2index[(chan)]
        #     pos_tos = [t for f, t in self.positive_edge_indices if f == self.hierarchy2index[(chan)]]
        #     self.negative_edge_indices_map[pos_from] = []
        #     for chan1 in range(0, 3):
        #         chan_to = self.hierarchy2index[(chan1)]
        #         if chan_to != pos_from:
        #             neg_to = [t for f, t in self.positive_edge_indices if f == self.hierarchy2index[(chan_to)]]
        #             neg_tos.extend(neg_to)
        #
        #     for clazz in range(0, 4):
        #         # self.positive_edge_indices.append((self.hierarchy2index[(chan)],
        #         #                                    self.hierarchy2index[(chan, clazz)]))
        #         pos_from = self.hierarchy2index[(chan)]
        #         pos_to = self.hierarchy2index[(chan, clazz)]
        #
        #         for t in range(0, 10):
        #             # self.positive_edge_indices.append((self.hierarchy2index[(chan, clazz)],
        #             #                                    self.hierarchy2index[(chan, clazz, t)]))

        # for i_from in range(0, self.num_embeddings):
        #     for i_to in range(0, self.num_embeddings):
        #         if not (i_from, i_to) in self.positive_edge_indices and i_from != i_to:
        #             self.negative_edge_indices.append((i_from, i_to))

    def __len__(self):
        return len(self.enhanced_mnist_samples)

    def __getitem__(self, i):
        return i

    def do_color_enhancement(self, mnist_samples):
        enhanced_mnist_samples = []
        for img, target in tqdm(mnist_samples):
            # add random color for each channel
            # for chan in range(0, 3):
            #     for clazz in range(0, 4):

            chan = random.randint(0, 2)
            clazz = random.randint(0, 3)
            # binarize
            img_np = np.array(img)
            img_np[img_np < 128] = 0
            img_np[img_np >= 128] = 255

            # convert to 3 channels
            img_np = np.tile(img_np[..., np.newaxis], [1, 1, 3])

            clazz_rand = np.random.randint(0, 64)
            new_val = clazz_rand + clazz * 64
            new_color = np.zeros([3])
            new_color[chan] = new_val

            img_np[(img_np == 255).all(axis=-1)] = new_color
            img_np = np.swapaxes(img_np, 0, -1)
            all_target = chan * 40
            all_target += clazz * 10
            all_target += target
            enhanced_mnist_samples.append([img_np, chan, clazz, target, all_target])
        return enhanced_mnist_samples

    def __getitem__(self, i):
        image, chan, clazz, label, network_target = self.enhanced_mnist_samples[i]

        output = {"image": image.astype(np.float32) / 255.,
                  "target": network_target,
                  "image_org": image,
                  "parent_hierarchy": (chan, clazz, label),
                  "lvl1_idx": self.hierarchy2index[(chan, clazz)],
                  "lvl0_idx": self.hierarchy2index[(chan,)],
                  "parent_idx": self.hierarchy2index[(chan, clazz, label)]}
        return output



class ColorMNIST_old(Dataset):

    def __init__(self, datapath, split, subsample_num=10):
        """
        mnist with hierarchy
        hierarchy consists of colors to create a visual hierarchy.

        red   0.. 64 => up_red_0 => red
        red  65..128 => up_red_0 => red
        red 129..184 => up_red_1
        red 185..256 => up_red_1
        etc.
        for each digit
        no mixed colors

        :param datapath:
        :param split:
        :param subsample_num:
        """
        np.random.seed(42)
        self.split = split
        fname = "color_mnist_" + self.split + ".pkl"
        color_mnist_datapath = Path(datapath) / fname

        self.hierarchy_to_index = {}

        # nodes
        i = 0
        for chan in range(0, 3):  # red green blue
            self.hierarchy_to_index[(chan)] = i
            i += 1
            for clazz in range(0, 4):  # different shades of red green blue
                self.hierarchy_to_index[(chan, clazz)] = i
                i += 1
                for t in range(0, 10):  # different digits
                    self.hierarchy_to_index[(chan, clazz, t)] = i
                    i += 1
        self.index_to_hierarchy = {v: k for k, v in self.hierarchy_to_index.items()}
        self.num_embeddings = i

        self.positive_edges = []
        for chan in range(0, 3):
            for clazz in range(0, 4):
                self.positive_edges.append((self.hierarchy_to_index[(chan)],
                                            self.hierarchy_to_index[(chan, clazz)]))
                for t in range(0, 10):
                        self.positive_edges.append((self.hierarchy_to_index[(chan, clazz)],
                                                    self.hierarchy_to_index[(chan, clazz, t)]))
        self.negative_edges_from = {'all': {}, '0': {}, '1': {}}
        self.negative_edges_to = {'all': {}, '0': {}, '1': {}}

        self.class_embeddings = []
        self.hierarchy_to_class = {}
        counter = 0
        for chan in range(0,3):
            for clazz in range(0,4):
                for t in range(0, 10):
                    self.class_embeddings.append(self.hierarchy_to_index[(chan, clazz, t)])
                    self.hierarchy_to_class[(chan, clazz, t)] = counter
                    counter += 1

        for chan0 in range(0, 3):
            for chan1 in range(0, 3):
                # vertical
                edge = (self.hierarchy_to_index[chan0], self.hierarchy_to_index[chan1])
                if edge not in self.positive_edges and edge[0] != edge[1]:
                    self.negative_edges_from['all'][edge[0]] = self.negative_edges_from['all'].get(edge[0], []) + [edge]
                    self.negative_edges_from['0'][edge[0]] = self.negative_edges_from['0'].get(edge[0], []) + [edge]

                for clazz0 in range(0, 4):
                    # from chan to wrong chan's clazz:
                    edge = (self.hierarchy_to_index[chan0], self.hierarchy_to_index[(chan1, clazz0)])
                    if edge not in self.positive_edges and edge[0] != edge[1]:
                        self.negative_edges_from['all'][edge[0]] = self.negative_edges_from['all'].get(edge[0], []) + [edge]
                        self.negative_edges_from['0'][edge[0]] = self.negative_edges_from['0'].get(edge[0], []) + [edge]

                    # vice versa
                    edge = (edge[1], edge[0])
                    if edge not in self.positive_edges and edge[0] != edge[1]:
                        self.negative_edges_from['all'][edge[0]] = self.negative_edges_from['all'].get(edge[0], []) + [edge]
                        self.negative_edges_from['0'][edge[0]] = self.negative_edges_from['0'].get(edge[0], []) + [edge]

                    for clazz1 in range(0, 4):
                        for t in range(0, 10):
                            # from all chan, clazz to all wrong chan, clazz, ts
                            edge = (self.hierarchy_to_index[(chan0, clazz0)], self.hierarchy_to_index[(chan1, clazz1, t)])
                            if edge not in self.positive_edges and edge[0] != edge[1]:
                                self.negative_edges_from['all'][edge[0]] = self.negative_edges_from['all'].get(edge[0], []) + [edge]
                                self.negative_edges_from['1'][edge[0]] = self.negative_edges_from['1'].get(edge[0], []) + [edge]

        # all negative parents for leaf node:
        self.negative_leaf_edges = {}
        self.negative_leaf_edges_indices = {}
        all_final_leafs = []
        for chan in range(0, 3):
            for clazz in range(0, 4):
                for t in range(0, 10):
                    all_final_leafs.append((chan, clazz, t))
        for chan in range(0, 3):
            for clazz in range(0, 4):
                for t in range(0, 10):
                    # add all but the current to the dict for quick lookup
                    self.negative_leaf_edges[(chan, clazz, t)] = [(ch, cl, tt) for ch, cl, tt in all_final_leafs if ch != chan and cl != clazz and t != tt]
                    self.negative_leaf_edges_indices[self.hierarchy_to_index[(chan, clazz, t)]] = [self.hierarchy_to_index[(ch, cl, tt)] for ch, cl, tt in all_final_leafs if ch != chan and cl != clazz and t != tt]
        if not color_mnist_datapath.is_file():

            self.mnist = torchvision.datasets.MNIST(datapath, train=self.split == 'train', download=True)

            # we will read all mnist first to have a deterministic blocking of leaf nodes
            t_start = datetime.now()
            mnist_samples = [[img, target] for img, target in self.mnist]
            self.enhanced_mnist_samples = self.do_color_enhancement(mnist_samples)
            pickle.dump(self.enhanced_mnist_samples, open(color_mnist_datapath, "wb"))
            print("done sampling, took", datetime.now() - t_start)
        else:
            self.enhanced_mnist_samples = pickle.load(open(color_mnist_datapath, "rb"))
        self.use_leaf_label = [1 for i in range(len(self.enhanced_mnist_samples))]
        if subsample_num > 1:
            for i in range(0, len(self.enhanced_mnist_samples)):
                self.use_leaf_label[i] = 0
                if i % subsample_num == 0:
                    self.use_leaf_label[i] = 1

    def do_color_enhancement(self, mnist_samples):
        enhanced_mnist_samples = []
        for img, target in mnist_samples:
            # add random color for each channel
            for chan in range(0, 3):
                for clazz in range(0, 4):
                    # binarize
                    img_np = np.array(img)
                    img_np[img_np < 128] = 0
                    img_np[img_np >= 128] = 255

                    # convert to 3 channels
                    img_np = np.tile(img_np[..., np.newaxis], [1, 1, 3])

                    clazz_rand = np.random.randint(0, 64)
                    new_val = clazz_rand + clazz * 64
                    new_color = np.zeros([3])
                    new_color[chan] = new_val

                    img_np[(img_np == 255).all(axis=-1)] = new_color
                    img_np = np.swapaxes(img_np, 0, -1)
                    all_target = chan * 40
                    all_target += clazz * 10
                    all_target += target
                    enhanced_mnist_samples.append([img_np, chan, clazz, target, all_target])
        return enhanced_mnist_samples

    def __len__(self):
        return len(self.enhanced_mnist_samples)

    def __getitem__(self, i):
        image, chan, clazz, label, all_target = self.enhanced_mnist_samples[i]

        if self.split == "train":
            if self.use_leaf_label[i] == 0:
                label = -1  # torch.randint(0, 62, [1]).item()
        # image = image.repeat([3, 1, 1])

        # sample positive and negative edge
        # positive_edge = self.positive_edges[i % len(self.positive_edges)]
        # negative_edge = self.negative_edges[i % len(self.negative_edges)]


        # negative edge:
        negative_parents = self.negative_leaf_edges_indices[self.hierarchy_to_index[(chan, clazz, label)]]#self.negative_leaf_edges[(chan, clazz, label)]
        random.shuffle(negative_parents)
        negative_parents = torch.Tensor(negative_parents)
        output = {"image": image.astype(np.float32) / 255.,
                  "image_org": image,
                  "label": all_target,
                  "label2": self.hierarchy_to_class[(chan, clazz, label)],
                  "negative_parents": negative_parents,
                  # "negative_edge": negative_edge,
                  # "positive_edge": positive_edge,
                  "hierarchy": (chan, clazz, label),
                  "hierarchy_embedding_index": self.hierarchy_to_index[(chan, clazz, label)],
                  "use_leaf_label": self.use_leaf_label[i]}

        # output = {
        #           "image_org": image,
        #           "label": label,
        #
        #           "use_leaf_label": self.use_leaf_label[i]}
        return output


class CombinedMnistDatasets(Dataset):
    # get data from here: https://github.com/renmengye/few-shot-ssl-public
    def __init__(self, datapath, split, subsample_num=10):
        # load digit datasets from torchvision
        self.split = split
        self.kmnist = torchvision.datasets.KMNIST(datapath, download=True)
        self.mnist = torchvision.datasets.MNIST(datapath, download=True)
        self.fashion_mnist = torchvision.datasets.FashionMNIST(datapath, download=True)
        self.emnist = torchvision.datasets.EMNIST(datapath, split="byclass", download=True, train=split == 'train')
        self.use_leaf_label = [1 for i in range(len(self.emnist))]

        _all_classes = set(list(string.digits + string.ascii_letters))
        all_classes_list = sorted(list(_all_classes))
        self.num2str = {k: v for k, v in enumerate(all_classes_list)}
        # for self.emnist
        self.label2hierarchy = {}

        # digit
        for l in range(0, 10):
            self.label2hierarchy[l] = 0
        # capital letter
        for l in range(10, 36):
            self.label2hierarchy[l] = 1
        # small letter
        for l in range(36, 62):
            self.label2hierarchy[l] = 2

        if subsample_num > 1:
            for i in range(0, len(self.emnist)):
                self.use_leaf_label[i] = 0
                if i % subsample_num == 0:
                    self.use_leaf_label[i] = 1

        print("done")

    def __len__(self):
        return len(self.emnist)

    def __getitem__(self, i):
        image, label = self.emnist.__getitem__(i)

        image = np.swapaxes(image, 1, 0)
        image = torchvision.transforms.functional.to_tensor(image)
        if self.split == "train":
            if self.use_leaf_label[i] == 0:
                label = -1  # torch.randint(0, 62, [1]).item()


        return output
    # def __init__(self, datapath, split, subsample_num=10):
    #     # self.download_and_extract_miniImagenet(datapath)
    #     datapath += "/mini-imagenet"
    #     self.create_wnid2num_mapping(datapath)
    #
    #
    #     csv_path = osp.join(datapath, 'train.csv')
    #     lines_all = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    #
    #     csv_path = osp.join(datapath, 'val.csv')
    #     lines_all.extend([x.strip() for x in open(csv_path, 'r').readlines()][1:])
    #
    #     csv_path = osp.join(datapath, 'test.csv')
    #     lines_all.extend([x.strip() for x in open(csv_path, 'r').readlines()][1:])
    #
    #     lines_test = lines_all[0::10]
    #     del lines_all[0::10]
    #     lines_val = lines_all[0::10]
    #     del lines_all[0::10]
    #     lines_train = lines_all
    #     if split == 'train':
    #         lines = lines_train
    #         subsample = True
    #     elif split == 'val':
    #         lines = lines_val
    #         subsample = False
    #     elif split == 'test':
    #         lines = lines_test
    #         subsample = False
    #     data = []
    #     label = []
    #     synset = []
    #     use_leaf_label = []
    #     lb = -1
    #
    #     self.wnids = []
    #     for i, l in enumerate(lines):
    #         name, wnid = l.split(',')
    #         path = osp.join(datapath, 'images', name)
    #         # if wnid not in self.wnids:
    #         #     self.wnids.append(wnid)
    #         #     lb += 1
    #         if i % subsample_num == 0 and subsample:
    #             use_leaf_label.append(1)
    #             data.append(path)
    #             label.append(self.wnid2num[wnid])
    #             # synset.append(self.get_synset_from_nltk(wnid))
    #             synset.append(wnid)
    #         # elif subsample:
    #         #     use_leaf_label.append(0)
    #         elif not subsample:
    #             use_leaf_label.append(1)
    #             data.append(path)
    #             label.append(self.wnid2num[wnid])
    #             # synset.append(self.get_synset_from_nltk(wnid))
    #             synset.append(wnid)
    #
    #     self.data = data
    #
    #     missing_labels = []
    #     for i in range(100):
    #         if not i in label:
    #             missing_labels.append(i)
    #     print("missing_labels", missing_labels)
    #     print("number of " + str(split) + " samples:", len(self.data), "of", len(lines))
    #     self.label = label
    #     self.synset = synset
    #     self.use_leaf_label = use_leaf_label
    #
    #     self.transform = transforms.Compose([
    #         transforms.Resize(84),
    #         transforms.CenterCrop(84),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    #     ])
    #     self.transfrom_only_resize = transforms.Compose([transforms.Resize(84),
    #         transforms.CenterCrop(84),
    #         transforms.ToTensor()])
    #
    # def get_synset_from_nltk(self, wnid):
    #     synset = wordnet.synset_from_pos_and_offset('n', int(wnid[1:]))
    #     return synset
    #
    #
    # def find_all_hypernyms(self, synset):
    #     # # https://stackoverflow.com/questions/42004286/get-a-full-list-of-all-hyponyms
    #     # relative = synset
    #     # hypos = lambda s: s.hyponyms()
    #     #
    #     # print(list(relative.closure(hypos)))
    #
    #     hypernyms = []
    #     h = synset.hypernyms()[0]
    #     while True:
    #         hypernyms.append(h)
    #         # TODO(nik): check this shit.
    #         try:
    #             h = h.hypernyms()[0]
    #         except:
    #             break
    #     return hypernyms
    #
    # def create_wnid2num_mapping(self, datapath):
    #     # also create num2stringlabel
    #     csv_path = osp.join(datapath, 'train.csv')
    #     lines_train = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    #     csv_path = osp.join(datapath, 'test.csv')
    #     lines_test = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    #     csv_path = osp.join(datapath, 'val.csv')
    #     lines_val = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    #
    #     all_lines = []
    #     all_lines.extend(lines_train)
    #     all_lines.extend(lines_test)
    #     all_lines.extend(lines_val)
    #
    #     self.wnid2num = {}
    #     self.num2wnid = {}
    #     self.num2str = {}
    #     counter = 0
    #     for l in all_lines:
    #         name, wnid = l.split(',')
    #
    #         if not wnid in self.wnid2num.keys():
    #             self.wnid2num[wnid] = counter
    #             self.num2wnid[counter] = wnid
    #             self.num2str[counter] = self.get_synset_from_nltk(wnid).lemmas()[0].name()
    #             counter += 1
    #
    #
    #     print("enumerated", counter, "classes")
    #
    # def download_and_extract_miniImagenet(self, root):
    #     from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
    #
    #     ## download miniImagenet
    #     # url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    #     file_id = '1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk'#'16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    #     filename = 'mini-Imagenet.tgz'
    #     download_file_from_google_drive(file_id, root, filename)
    #     fpath = os.path.join(root, filename)  # this is what download_file_from_google_drive does
    #     ## extract downloaded dataset
    #     from_path = os.path.expanduser(fpath)
    #     extract_archive(from_path)
    #
    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, i):
    #     path, label, synset, use_leaf_label = self.data[i], self.label[i], self.synset[i], self.use_leaf_label[i]
    #     image_org = Image.open(path).convert('RGB')
    #     image = self.transform(image_org)
    #     image_org = self.transfrom_only_resize(image_org)
    #     output = {"image": image,
    #               "image_org": image_org,
    #               "label": label,
    #               "synset": synset,
    #               "use_leaf_label": use_leaf_label}
    #     return output
