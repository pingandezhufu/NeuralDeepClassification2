# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import six.moves.cPickle
from six.moves import range
from numpy import *
coarse_labels=['aquatic_mammals','fish','flowers','food_containers','fruit_and_vegetables','household_electrical_devices','household_furniture','insects','large_carnivores','large_man-made_outdoor_things','large_natural_outdoor_scenes','large_omnivores_and_herbivores','medium_mammals','non-insect_invertebrates','people','reptiles','small_mammals','trees','vehicles_1','vehicles_2']
# for i in range(len(coarse_labels)):
#     print(i)
#     print(coarse_labels[i])
# 0
# aquatic_mammals
# 1
# fish
# 2
# flowers
# 3
# food_containers
# 4
# fruit_and_vegetables
# 5
# household_electrical_devices
# 6
# household_furniture
# 7
# insects
# 8
# large_carnivores
# 9
# large_man-made_outdoor_things
# 10
# large_natural_outdoor_scenes
# 11
# large_omnivores_and_herbivores
# 12
# medium_mammals
# 13
# non-insect_invertebrates
# 14
# people
# 15
# reptiles
# 16
# small_mammals
# 17
# trees
# 18
# vehicles_1
# 19
# vehicles_2


def load_batch(fpath, label_key='labels'):
    print("loading batch")
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = six.moves.cPickle.load(f)
    else:
        d = six.moves.cPickle.load(f, encoding="bytes")
        # decode utf8
        for k, v in d.items():
            del(d[k])
            d[k.decode("utf8")] = v

    f.close()
    data = d["data"]
    dataList=data.tolist()
    labels = d[label_key]
    print(label_key)
    outer_labels=d["coarse_labels"]

    ##new unique labels

    print len(dataList)
    print len(outer_labels)
    print len(labels)
    i=0
    while i < len(outer_labels)/2:
         #print(i)
         #print(outer_labels[i])
         if (outer_labels[i]!=14):
             #print(outer_labels[i])
             dataList.remove(dataList[i])
             del(labels[i])
             del(outer_labels[i])
             i=i-1
         i=i+1
    # temp_labels=labels
    # temp_coarselabels=outer_labels
    # temp_dataList=dataList
    while i < len(outer_labels):
         #print(i)
         #print(outer_labels[i])
         if (outer_labels[i]!=14):
             #print(outer_labels[i])
             dataList.remove(dataList[i])
             del(labels[i])
             del(outer_labels[i])
             i=i-1
         i=i+1
    j=0

    print len(outer_labels)
    print len(labels)
    new_labels=set()
    for i in range (len(labels)):
        new_labels.add(labels[i])
    print(len(new_labels))
    for i in range (len(labels)):
        labels_backup=new_labels.copy()
        j=0
        while len(labels_backup)>0:
            if labels[i]==labels_backup.pop():
                labels[i]=j+10
                #print(j)
            j=j+1
            #print(j)
    data = asarray(dataList)
    print(len(dataList))
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
