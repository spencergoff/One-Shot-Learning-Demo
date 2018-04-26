#!/usr/bin/python
#   M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
#   International Conference on Pattern Recognition, pp. 566-568.

import os
import boto3
from PIL import Image
import numpy as np
from scipy.ndimage import imread
from scipy.spatial.distance import cdist

nrun = 1 #Number of classification runs
path_to_script_dir = '/tmp/' #os.path.dirname(os.path.realpath(__file__)) #gets the directory of this file
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
fname_label = 'class_labels.txt'  # Where class labels are stored for each run
classification_result = '' #this is the result of an image being classified
testing_location = '/tmp/all_runs/run01/test/search.jpg' #location of the search image within the Lambda directory
train_files = [] #user's collection
test_files = [] #image to search/classify

def lambda_handler(event, context):
    print 'Starting lambda_handler'
    global classification_result
    global train_files
    global test_files
    global testing_location

    s3 = boto3.resource('s3')
    user = event['user']
    print 'user: ' + user
    bucket = 'whatsteddysname-userfiles-mobilehub-463317519'
    collection = s3.Bucket(bucket)
    i = 0

    #delete any existing contents of /temp/ training
    if os.path.exists('/tmp/all_runs/run01/training/'):
        for file in os.listdir('/tmp/all_runs/run01/training/'):
            filename = '/tmp/all_runs/run01/training/' + str(file)
            print('removing from tmp: ' + filename)
            os.remove(filename)

    #delete any existing contests of /tmp/ searching
    if os.path.exists('/tmp/all_runs/run01/test/'):
        for file in os.listdir('/tmp/all_runs/run01/test/'):
            filename = '/tmp/all_runs/run01/test/' + str(file)
            print('removing from tmp: ' + filename)
            os.remove(filename)

    #create the necessary directories in tmp
    if not os.path.exists('/tmp/all_runs/run01/training/'):
        os.makedirs('/tmp/all_runs/run01/training/') #creates a directory in tmp (only exists during runtime) for the user's collection
    if not os.path.exists('/tmp/all_runs/run01/test/'):
        os.makedirs('/tmp/all_runs/run01/test/') #creates a directory in tmp (only exists during runtime) for the search image

    path_to_script_dir = '/tmp/' #os.path.dirname(os.path.realpath(__file__)) #gets the directory of this file
    path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
    file_t = open('/tmp/all_runs/run01/class_labels.txt', 'w+') #creates the file

    i = 0
    training_location = ''
    train_files = []
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #downloads the user's COLLECTION (don't confuse with search)
    for image in collection.objects.filter(Prefix='public/{}/gray'.format(user)):
        if str(image.key).endswith('.jpg'):
            training_location = '/tmp/all_runs/run01/training/{}.jpg'.format(i)
            if training_location not in train_files:
                for f in train_files:
                    print('train_file: ' + str(f))
                print('appending ' + str(image.key) + ' to train_files as ' + training_location)
                train_files.append(str(training_location))
                s3.meta.client.download_file(bucket, image.key, training_location)
                if os.path.isfile(training_location):
                    print('Downloaded ' + image.key + ' to ' + training_location)
                    i = i + 1
                else:
                    print('Failed to download ' + image.key + ' for training')

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('len(train_files) immediately after download: ' + str(len(train_files)))
    for f in train_files:
        print('train_file: ' + str(f))

    test_files = []
    #downloads the image(s) to SEARCH (don't confuse with collection)
    for image in collection.objects.filter(Prefix='public/search-images/{}'.format(user)):
        if image.key == '/' or str(image.key).endswith('example-user-1.jpg') or str(image.key).endswith('example-user-2.jpg') \
         or str(image.key).endswith('example-user-3.jpg'): #example-user-_.jpg are for testing purposes
            print('continuing...')
            continue
        s3.meta.client.download_file(bucket, image.key, testing_location) #downloads the image to be classified (searched)
        if str(image.key).endswith('.jpg') and testing_location not in test_files:
            print('appending ' + str(image.key) + 'to test_files')
            test_files.append(testing_location)
        if os.path.isfile(testing_location):
            print('Downloaded ' + image.key + ' to ' + testing_location)
        else:
            print('Failed to download ' + image.key + 'for searching')

    file_t.write('/tmp/all_runs/run01/test/search.jpg /tmp/all_runs/run01/training/01.jpg')
    file_t.close()
    file = open('/tmp/all_runs/run01/class_labels.txt', 'r+') #opens file for reading and writing

    print('One-shot classification with Modified Hausdorff Distance')
    perror = np.zeros(nrun)
    for r in range(nrun):
        perror[r] = classification_run('run{:02d}'.format(r + 1),
                                       load_img_as_points,
                                       modified_hausdorf_distance,
                                       'cost')
    total = np.mean(perror)

    file.close()
    real_classifcation_result = ''
    index_of_classification_in_lambd = -1
    classification_result = str(classification_result)

    #find out the index of classification_result within Lambda
    for x in classification_result.split('/'):
        for y in x.split('.'):
            print('y = ' + y)
            if y.isdigit():
                index_of_classification_in_lambd = int(y)
                break

    #iterate through the s3 bucket until i = index_of_classification_in_lambd, then return that image's path
    for i, image in enumerate(collection.objects.filter(Prefix='public/{}'.format(user))):
        if i == index_of_classification_in_lambd+1:
            real_classifcation_result = str(image.key)
            if user != 'example-user': #for testing purposes
                path = 'public/search-images/{}.jpg'.format(user)
                s3.Object(bucket, path).delete() #deletes the image from the seach directory in s3


    print('THE CLASSIFICATION IS: ' + real_classifcation_result)
    return real_classifcation_result

def classification_run(folder, f_load, f_cost, ftype='cost'):
    # Compute error rate for one run of one-shot classification
    #
    # Input
    #  folder : contains images for a run of one-shot classification
    #  f_load : itemA = f_load('file.png') should read in the image file and
    #           process it
    #  f_cost : f_cost(itemA,itemB) should compute similarity between two
    #           images, using output of f_load
    #  ftype  : 'cost' if small values from f_cost mean more similar,
    #           or 'score' if large values are more similar
    #
    # Output
    #  perror : percent errors (0 to 100% error)
    #
    assert ftype in {'cost', 'score'}
    global classification_result
    global train_files
    global test_files
    global testing_location

    print('made it to classification_run')

    with open(os.path.join(path_to_all_runs, folder, fname_label)) as f:
        pairs = [line.split() for line in f.readlines()]

    # Unzip the pairs into two sets of tuples
    #test_files, train_files = zip(*pairs)

    answers_files = list(train_files)  # Copy the training file list
    test_files = sorted(test_files)
    train_files = sorted(train_files)
    n_train = len(train_files)
    n_test = len(test_files)

    print('len(test_files): ' + str(len(test_files)))
    print('len(train_files): ' + str(len(train_files)))
    for f in train_files:
        print('train_file: ' + str(f))

    train_items = [f_load(os.path.join(path_to_all_runs, f))
                   for f in train_files]
    for file in test_files:
        print('test_files file: ' + str(file)) #Ex. output: test_files file: public/search-images/example-user.jpg

    testing_location_array = []
    testing_location_array.append('/tmp/all_runs/run01/test/search.jpg')
    test_items = [f_load(os.path.join(path_to_all_runs, f)) #path_to_all_runs = /tmp/all_runs
                  for f in testing_location_array] #test_file entry: public/search-images/example-user.jpg
                  #note: testing_location = '/tmp/all_runs/run01/test/search.jpg'

    #print('train_items: ' + str(train_items))
    #print('test_items: ' + str(test_items))

    # Compute cost matrix
    i = 0
    costM = np.zeros((n_test, n_train))
    for i, test_i in enumerate(test_items):
        for j, train_j in enumerate(train_items):
            print('adding to costM')
            costM[i, j] = f_cost(test_i, train_j)
            print('costM[i, j]' + str(costM[i, j]))
    if ftype == 'cost':

        y_hats = np.argmin(costM, axis=1) #the indices of the smallest values in each row; axis 1 runs horizonally accross columns
    elif ftype == 'score':
        y_hats = np.argmax(costM, axis=1)
    else:
        # This should never be reached due to the assert above
        raise ValueError('Unexpected ftype: {}'.format(ftype))

    for y_hat, answer in zip(y_hats, answers_files):
        classification_result = str(train_files[y_hat])

    # compute the error rate by counting the number of correct predictions
    correct = len([1 for y_hat, answer in zip(y_hats, answers_files)
                   if train_files[y_hat] == answer])

    pcorrect = correct / float(n_test)  # Python 2.x ensure float division
    perror = 1.0 - pcorrect
    return perror * 100

def modified_hausdorf_distance(itemA, itemB):
    # Modified Hausdorff Distance
    #
    # Input
    #  itemA : [n x 2] coordinates of black pixels
    #  itemB : [m x 2] coordinates of black pixels
    #
    #  M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
    #  International Conference on Pattern Recognition, pp. 566-568.
    #
    print('calculating distance...')
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)


def load_img_as_points(filename):
    # Load image file and return coordinates of black pixels in the binary image
    #
    # Input
    #  filename : string, absolute path to image
    #
    # Output:
    #  D : [n x 2] rows are coordinates
    #
    print('loading image as points: ' + str(filename))
    I = imread(filename, flatten=True)
    # Convert to boolean array and invert the pixel values
    I = ~np.array(I, dtype=np.bool)
    # Create a new array of all the non-zero element coordinates
    D = np.array(I.nonzero()).T
    return D - D.mean(axis=0)
