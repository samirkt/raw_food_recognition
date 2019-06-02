#!/usr/bin/env python

import sys
import os
import urllib2
import pandas as pd


### Download all image urls into data/train/
###     and store used urls in data/url_map/

# Extract Imagenet IDs for each image class
ingredients = open('ingredients_list.txt')
name,grp = [],[]
for line in ingredients:
    l = line.strip().split(',')
    name.append(l[0])
    grp.append(l[1])
ingredients.close()

# Read URLs and group by class ID
fp = open('urls.txt','r')
urls = [line.split(',',2) for line in fp]
urls = pd.DataFrame(urls,columns=['id','num','url'])
grps = urls.groupby(['id'])
fp.close()

# Manually omit items from ingredients_list.txt (i.e. already downloaded):
#           ...store (row number - 1) in array below
skip = []
#skip.extend(range(77))
#skip.remove(66)
#skip.remove(37)

# Download all URLs for each image class
for i in range(len(grp)):
    # Skip image class if specified
    if i in skip:
        continue
    
    # Get class-specific URLs
    item = grps.get_group(grp[i])

    print(name[i] + ': ' + str(len(item)))

    # Create needed directories
    url_dir = 'data/url_map/'
    data_dir = 'data/train/'+name[i]+'/'
    if not os.path.exists(url_dir):
        os.makedirs(url_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Open URL map file for class
    url_map = open(url_dir+name[i]+'.txt','wb')

    # Download each URL
    valid = 0
    invalid = 0
    count = 0
    for ind,row in item.iterrows():
        try:
            # Download and save
            resource = urllib2.urlopen(row['url'],timeout=3)
            obj_name = name[i] + str(count)
            output = open(data_dir+obj_name+'.jpg','wb')
            output.write(resource.read())
            output.close()

            url_map.write(obj_name+','+row['url'])
            
            # Count successful downloads
            valid += 1
            count += 1
        except:
            # Count unsuccessful downloads
            invalid += 1
            count += 1

        # Print class-specific progress
        sys.stdout.write('\r\t'+ str(valid) + '/' + str(count))
        sys.stdout.flush()


    # Print overall progress
    sys.stdout.write('\r\t'+ str(valid) + '/' + str(valid+invalid) + '\n')
    url_map.close()
