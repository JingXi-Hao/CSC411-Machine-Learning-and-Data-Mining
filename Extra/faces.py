
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread, imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import os.path
import shutil


#act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

act_test = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

# define best_theta as global variable
best_theta = np.array([[]])
theta_largest_set = np.array([[]])
theta_new_way = np.array([[]])



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()      


#Note: you need to create the uncropped folder first in order 
#for this to work

# part 1
def process_actor_image(dir):
    #  make working directory
    if (not os.path.exists("cropped/")):
        os.mkdir("cropped/")
        
    if (not os.path.exists("uncropped/")):
        os.mkdir("uncropped/")
    
    #"facescrub_actors.txt"   "facescrub_actresses.txt"
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dir): # change the starter code here
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                else:
                    try:
                        # read the corresponding image with the filename from
                        # uncropped folder
                        image = imread("uncropped/"+filename)
                        # assign value to x1,y1,x2,y2
                        x1, y1, x2, y2 = line.split()[5].split(',')
                        # convert to integer
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        # crop out the face
                        face = image[y1:y2, x1:x2]
                        # convert image to grayscale
                        gray_face = rgb2gray(face)
                        # resize the image to 32x32
                        resized_face = imresize(gray_face, [32, 32])
                        # put the copped out image into cropped folder
                        imsave("cropped/"+filename, resized_face)
                    except:
                        continue
                        
                print filename
                i += 1
                
# simply copy the code from process_actor_image() and change the input file
# directory
# read in data for testing act_test
def process_act_test_image(dir):
    
    #  make working directory
    if (not os.path.exists("cropped_test/")):
        os.mkdir("cropped_test/")
        
    if (not os.path.exists("uncropped_test/")):
        os.mkdir("uncropped_test/")
    
    for a in act_test:
        name = a.split()[1].lower()
        i = 0
        for line in open(dir):
            #"facescrub_actors.txt"
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped_test/"+filename), {}, 30)
                if not os.path.isfile("uncropped_test/"+filename):
                    continue
                else:
                    try:
                        # read the corresponding image with the filename from
                        # uncropped folder
                        image = imread("uncropped_test/"+filename)
                        # assign value to x1,y1,x2,y2
                        x1, y1, x2, y2 = line.split()[5].split(',')
                        # convert to integer
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        # crop out the face
                        face = image[y1:y2, x1:x2]
                        # convert image to grayscale
                        gray_face = rgb2gray(face)
                        # resize the image to 32x32
                        resized_face = imresize(gray_face, [32, 32])
                        # put the copped out image into cropped folder
                        imsave("cropped_test/"+filename, resized_face)
                    except:
                        continue
                        
                print filename
                i += 1
                

# part 2
def separate_dataset():
    index = 0
    count = 0
    
    # run random.seed every time before run random.choice for reproducible
    np.random.seed(1)
    random_nums = np.random.choice(150, 120, replace=False)
    
    # create three folders
    if (not os.path.exists("training_set/")):
        os.mkdir("training_set/")
    
    if (not os.path.exists("validation_set/")):
        os.mkdir("validation_set/")

    if (not os.path.exists("test_set/")):
        os.mkdir("test_set/")
        
    # get all names of contents in "cropped" folder and store them in a list
    contents_filename = os.listdir("cropped/")
    
    if ('.DS_Store' in contents_filename):
        contents_filename.remove('.DS_Store')
        
    # create folders with each actor/actress name inside the "training set" 
    # folder
    for full_filename in contents_filename:
        # process filename
        last_name_with_num, extension = full_filename.split('.')
        index = len(last_name_with_num) - 1
        
        while (last_name_with_num[index].isdigit()):
            index = index - 1
            
        last_name = last_name_with_num[:index+1]
        num = int(last_name_with_num[index+1:])
        
        # create folder with last_name
        if (not os.path.exists("training_set/" + last_name + "/")):
            os.mkdir("training_set/"+ last_name + "/")
            
        if (not os.path.exists("validation_set/" + last_name + "/")):
            os.mkdir("validation_set/" + last_name + "/")
        
        if (not os.path.exists("test_set/" + last_name + "/")):
            os.mkdir("test_set/" + last_name + "/")
            
        # add to corresponding folder
        temp = np.where(random_nums[:] == num)[0]
        if (temp.size != 0):
            position = temp[0]
            if (0 <= position < 100):
                shutil.copy("cropped/"+ full_filename, 
                            "training_set/" + last_name + "/" + full_filename)
            elif (100 <= position < 110):
                shutil.copy("cropped/"+ full_filename, 
                            "validation_set/" + last_name + "/" + full_filename)
            elif (110 <= position < 120):
                shutil.copy("cropped/"+ full_filename, 
                            "test_set/" + last_name + "/" + full_filename)
                        
# part 3
# the cost function
def f(x, y, theta):
    return (0.0025)*(sum((np.dot(x, theta) - y)**2))
    
# derivative of the cost function
def df(x, y, theta):
    return (0.005)*(np.dot(x.T, (np.dot(x, theta) - y)))
    
# gradient descent algorithm
def gradient_descent(f, df, x, y, init_theta, alpha):
    theta = init_theta.copy()
    max_iter = 50000
    count = 0
    
    while(count < max_iter):
        theta -= alpha*df(x, y, theta)
        count = count + 1
    return theta

# helper function for test dataset
def build_matrix(dir, x, y, classifier, size):
    i = 0
    one_row = np.array([[1]])
    zero_row = np.array([[0]])
    drescher = np.array([[1, 0, 0, 0, 0, 0]])
    ferrera = np.array([[0, 1, 0, 0, 0, 0]])
    chenoweth = np.array([[0, 0, 1, 0, 0, 0]])
    baldwin = np.array([[0, 0, 0, 1, 0, 0]])
    hader = np.array([[0, 0, 0, 0, 1, 0]])
    carell = np.array([[0, 0, 0, 0, 0, 1]])

    all_files = os.listdir(dir)
    if ('.DS_Store' in all_files):
        all_files.remove('.DS_Store')
    
    # create a widthxlength matrix where the first column is only 0/1 where 0 
    # represents carell and 1 indicates hader
    #for filename in all_files:
    for i in range(0, size):
        # read from list with the "file" name; convert to 2D
        image = imread(dir + all_files[i])
        
        # reshape the 2D array into 1x1024 vector
        vectored_image = np.reshape(image, (1, 1024))
        
        # add 1 into true_person list
        if (classifier.isdigit()):
            if (int(classifier) == 1):
                y = np.concatenate((y, one_row), axis= 1)
            elif (int(classifier) == 0):
                y = np.concatenate((y, zero_row), axis= 1)
        
        elif (classifier.isalpha()):
            if (classifier == "drescher"):
                y = np.concatenate((y, drescher), axis = 1)
            elif (classifier == "ferrera"):
                y = np.concatenate((y, ferrera), axis = 1)
            elif (classifier == "chenoweth"):
                y = np.concatenate((y, chenoweth), axis = 1)
            elif (classifier == "baldwin"):
                y = np.concatenate((y, baldwin), axis = 1)
            elif (classifier == "hader"):
                y = np.concatenate((y, hader), axis = 1)
            elif (classifier == "carell"):
                y = np.concatenate((y, carell), axis = 1)
                
        # add 1 at the beginning of the vector to save the constant term
        processed_row_image = np.concatenate((one_row, vectored_image), axis = 1)
        
        # add the i(th) row into train_set matrix
        x = np.concatenate((x, processed_row_image), axis = 1)

    return x, y
    
# helper function for test dataset
def count_success(estimate, size):
    index = 0
    count = 0
    for index in range(0, size):
        if (estimate[index] >= 0.5):
            if (index < (size/2)):
                count += 1 
        else:
            if(index >= (size/2)):
                count += 1
    return count
        
def test_dataset_for_hader_carell():
    # this array represents y
    train_set = np.array([[]])
    # this array represents x
    true_person = np.array([[]])
    init_theta = np.zeros((1025, 1))
    alpha = 9*1e-8
    recognized_faces = 0
    
    # call helper function build_matrix for hader first
    train_set, true_person = build_matrix("training_set/hader/", train_set,
                                        true_person, "1", 100)
                                        
    # call helper function build_matrix for carell
    train_set, true_person = build_matrix("training_set/carell/", train_set,
                                        true_person, "0", 100)
    
    
    train_set = np.reshape(train_set, (200, 1025))
    true_person = np.reshape(true_person, (200, 1))
    
    # call the gradient descent algorithm function
    global best_theta 
    best_theta = gradient_descent(f, df, train_set, true_person, init_theta, alpha)
    estimate = np.dot(train_set, best_theta)
    recognized_faces = count_success(estimate, 200)
    success_rate = recognized_faces/200.0
    
    
    # test validation set
    # this array represents y
    valid_set = np.array([[]])
    # this array represents x
    true_person2 = np.array([[]])
    recognized_faces = 0
    
    # call helper function build_matrix for hader first
    valid_set, true_person2 = build_matrix("validation_set/hader/", valid_set,
                                        true_person2, "1", 10)
                                        
    # call helper function build_matrix for carell
    valid_set, true_person2 = build_matrix("validation_set/carell/", valid_set,
                                        true_person2, "0", 10)
    
    valid_set = np.reshape(valid_set, (20, 1025))
    true_person2 = np.reshape(true_person2, (20, 1))
    
    # count the amount of correct predictions
    estimate = np.dot(valid_set, best_theta)
    recognized_faces = count_success(estimate, 20)
    success_rate2 = recognized_faces/20.0
    
    # print out the results
    print "The performance of the classifier on the training set is: ", success_rate
    print "The performance of the classifier on the validation set is: ", success_rate2
    
    print "\n"
    
    print "The value of cost function on the training set is: ", f(train_set, true_person, best_theta)
    print "The value of cost function on the validation set is: ", f(valid_set, true_person2, best_theta)*10
    
    
    # test test set
    # this array represents y
    test_set = np.array([[]])
    # this array represents x
    true_person = np.array([[]])
    recognized_faces = 0
    
    # call helper function build_matrix for hader first
    test_set, true_person = build_matrix("test_set/hader/", test_set,
                                        true_person, "1", 10)
                                        
    # call helper function build_matrix for carell
    test_set, true_person = build_matrix("test_set/carell/", test_set,
                                        true_person, "0", 10)
    
    test_set = np.reshape(test_set, (20, 1025))
    true_person = np.reshape(true_person, (20, 1))
    
    # count the amount of correct predictions
    estimate = np.dot(test_set, best_theta)
    recognized_faces = count_success(estimate, 20)
    success_rate = recognized_faces/20.0
    #print success_rate
    
# part 4
def view_full_theta():
    # view the theta obtained from the full training set
    # assign global variable best_theta first; if best_theta is empty, run 
    # test_dataset_for_hader_carell() first
    if (best_theta.size != 0):
        theta = np.reshape(best_theta[1:], (32, 32))
        fig = figure(1)
        ax = fig.gca()
        graph = ax.imshow(theta, cmap=cm.coolwarm)
        fig.colorbar(graph)
        title("Heatmap of Theta for Full Training Set")
        show()
    else:
        test_dataset_for_hader_carell()
        view_full_theta()
        
def view_partial_theta():
    # pick first two images from training set for hader and for carell
    # so, pick 4 images in total
    train_set = np.array([[]])
    true_person = np.array([[]])
    init_theta = np.zeros((1025,1))
    alpha = 4*1e-8
    
    train_set, true_person = build_matrix("training_set/hader/", train_set,
                                        true_person, "1", 2)
                                        
    train_set, true_person = build_matrix("training_set/carell/", train_set,
                                        true_person, "0", 2)
                                        
    train_set = np.reshape(train_set, (4, 1025))
    true_person = np.reshape(true_person, (4, 1))

    temp_theta = gradient_descent(f, df, train_set, true_person, init_theta, alpha)
    
    # check for accuracy of temp_theta computed
    estimate = np.dot(train_set, temp_theta)
    #print estimate
    recognized_faces = count_success(estimate, 4)
    #print recognized_faces
    success_rate = recognized_faces/4.0
    #print success_rate
    
    # view temp_theta
    theta_2 = np.reshape(temp_theta[1:], (32,32))
    fig_2 = figure(2)
    bx = fig_2.gca()
    graph_2 = bx.imshow(theta_2, cmap=cm.coolwarm)
    fig_2.colorbar(graph_2)
    title("Heatmap of Theta for Partial Training Set")
    show()
    
# part 5
def test_overfit():
    # overfitting: test with large amount of dataset which we choose all 
    # 100 images for each actor
    # define classifiers where 1 represents female and 0 represents male
    i = 10
    
    names = os.listdir("training_set/")
    init_t = np.zeros((1025, 1))
    #alpha = 3.476065*1e-8 # 3.476065 not work
    #alpha = 3.476066*1e-8
    #alpha = 3*1e-8
    alpha = 9*1e-9
    training_set_size = np.array([[]])
    train_perform = np.array([[]])
    #all_thetas = []
    valid_perform = np.array([[]])
    
    # create matrix for test validation set
    # test validation set using temp_theta computed
    valid_set = np.array([[]])
    true_person = np.array([[]])
    # build matrix first
    valid_set, true_person = build_matrix("validation_set/chenoweth/", valid_set, 
                                        true_person, "1", 10)
    valid_set, true_person = build_matrix("validation_set/drescher/", valid_set, 
                                        true_person, "1", 10)
    valid_set, true_person = build_matrix("validation_set/ferrera/", valid_set, 
                                        true_person, "1", 10)
    valid_set, true_person = build_matrix("validation_set/baldwin/", valid_set, 
                                        true_person, "0", 10)
    valid_set, true_person = build_matrix("validation_set/carell/", valid_set, 
                                        true_person, "0", 10)
    valid_set, true_person = build_matrix("validation_set/hader/", valid_set, 
                                        true_person, "0", 10)
    
    valid_set = np.reshape(valid_set, (60, 1025))
    true_person = np.reshape(true_person, (60, 1))
    
    while (i <= 100):
        
        # test training set and find theta of training set
        train_set = np.array([[]])
        true_person = np.array([[]])
        recognized_faces = 0
        
        training_set_size = np.append(training_set_size, 6*i)
        
        train_set, true_person = build_matrix("training_set/chenoweth/", train_set, 
                                            true_person, "1", i)
        train_set, true_person = build_matrix("training_set/drescher/", train_set, 
                                            true_person, "1", i)
        train_set, true_person = build_matrix("training_set/ferrera/", train_set, 
                                            true_person, "1", i)
        train_set, true_person = build_matrix("training_set/baldwin/", train_set, 
                                            true_person, "0", i)
        train_set, true_person = build_matrix("training_set/carell/", train_set, 
                                            true_person, "0", i)
        train_set, true_person = build_matrix("training_set/hader/", train_set, 
                                            true_person, "0", i)
                                            
        train_set = np.reshape(train_set, (6*i, 1025))
        #print train_set
        true_person = np.reshape(true_person, (6*i, 1))
        #print true_person
        
        temp_theta = gradient_descent(f, df, train_set, true_person, 
                                    init_t, alpha)
        
        # assign global variable
        if (i == 100):
            global theta_largest_set
            theta_largest_set = temp_theta
        
        estimate = np.dot(train_set, temp_theta)
        recognized_faces = count_success(estimate, 6*i)
        success_rate = recognized_faces/float(6*i)
        train_perform = np.append(train_perform, success_rate)
        print "test accuracy on training set: ", success_rate
        
        # test validation set using temp_theta computed above
        recognized_faces = 0
        estimate2 = np.dot(valid_set, temp_theta)
        recognized_faces = count_success(estimate2, 60)
        success_rate = recognized_faces/60.0
        valid_perform = np.append(valid_perform, success_rate)
        print "test accuracy on validation set: ", success_rate
        
        i += 10
    
    # plot the graph
    plt.plot(training_set_size, train_perform, marker="s", color="r", linestyle="--", label="training_set")
    plt.plot(training_set_size, valid_perform, marker="o", color="b", linestyle="--", label="validation_set")
    plt.axis([0, 700, 0, 1.5])
    plt.ylabel("Performance of Classifier")
    plt.xlabel("Size of Training Set")
    plt.title("Performance of Classifier on Training and Validation Set")
    plt.legend()
    plt.show()

# test performance of classifier for act_test
# Note: run test_overfit() first in order to get theta_largest_set assigned
def test_act_test():
    count = 0
    
    all_people = os.listdir("cropped_test/")
    if ('.DS_Store' in all_people):
        all_people.remove('.DS_Store')
    
    total = len(all_people)
        
    x = np.array([[]])
    y = np.array([[]])
    one_row = np.array([[1]])
    zero_row = np.array([[0]])
    
    for person in all_people:
        image_name = person.split(".")[0]
        index = len(image_name) - 1
        
        while (image_name[index].isdigit()):
            index = index - 1
            
        last_name = image_name[:index+1]
        num = int(image_name[index+1:])
        
        if (last_name == "butler" or last_name == "radcliffe" or last_name == "vartan"):
            y = np.concatenate((y, zero_row), axis = 1)
            image = imread("cropped_test/" + person)
            reshaped_image = np.reshape(image, (1, 1024))
            x1 = np.concatenate((one_row, reshaped_image), axis = 1)
            x = np.concatenate((x, x1), axis = 1)
            
        if (last_name == "harmon" or last_name == "gilpin" or last_name == "bracco"):
            y = np.concatenate((y, one_row), axis = 1)
            image = imread("cropped_test/" + person)
            reshaped_image = np.reshape(image, (1, 1024))
            x1 = np.concatenate((one_row, reshaped_image), axis = 1)
            x = np.concatenate((x, x1), axis = 1)
            
    x = np.reshape(x, (total, 1025))
    y = np.reshape(y, (total, 1))
    
    estimate = np.dot(x, theta_largest_set)
    
    for i in range(0, total):
        if (estimate[i] >= 0.5):
            if (y[i] == 1.0):
                count += 1
        else:
            if (y[i] == 0):
                count += 1
    
    success_rate = count/float(total)
    print success_rate
    
# part 6(c)
# cost function
def f2(x, y, theta):
    return sum((np.dot(theta.T, x) - y)**2)
    
# derivative of the cost function
def df2(x, y, theta):
    return 2*np.dot(x, (np.dot(theta.T, x) - y).T)
    
# gradient descent algorithm
def gradient_descent2(f2, df2, x, y, init_theta, alpha):
    theta = init_theta.copy()
    max_iter = 4000
    count = 0
    
    while(count < max_iter):
        theta -= alpha*df2(x, y, theta)
        #if (count == 49999):
            #print "gradient is ", df(train_set, true_person, theta), "\n"
        count = count + 1
    #print theta
    return theta

# part 6 (d)
# use theta generated from part 7, which means we have to run part 7 first to
# get the value of theta_new_way
def test_gradient(x, y):
    theta_copy = theta_new_way.T.copy()
    theta_copy2 = theta_new_way.T.copy()
    h = 0.000001
    df2_result = df2(x, y, theta_new_way.T)
    
    # use part 7 theta to test; therefore, run part 7 first
    for i in range(0, 4):
        for j in range (0, 4):
            theta_copy[i][j] = theta_copy[i][j] + h
            theta_copy2[i][j] = theta_copy2[i][j] - h
            diff = f2(x, y, theta_copy) - f2(x, y, theta_copy2)
            deriv = diff/(2*h)
            comparison = abs(deriv- df2_result[i][j])
            print "Difference of changing the cell row ", i, " and column ", j, ": ", comparison
            theta_copy[i][j] = theta_copy[i][j] - h
            theta_copy2[i][j] = theta_copy2[i][j] + h

# part 7
def test_dataset_new_way():
    x = np.array([[]])
    y = np.array([[]])
    x2 = np.array([[]])
    y2 = np.array([[]])
    success = 0
    success2 = 0
    init_theta = np.zeros((1025, 6))
    alpha = 4*1e-11
    
    # define classifiers:
    #    ---drescher: [1,0,0,0,0,0]
    #    ---ferrera: [0,1,0,0,0,0]
    #    ---chenoweth: [0,0,1,0,0,0]
    #    ---baldwin: [0,0,0,1,0,0]
    #    ---hader: [0,0,0,0,1,0]
    #    ---carell: [0,0,0,0,0,1]
    # build matrix for x and y by using helper function for the training set
    x, y = build_matrix("training_set/drescher/", x, y, "drescher", 100)
    x, y = build_matrix("training_set/ferrera/", x, y, "ferrera", 100)
    x, y = build_matrix("training_set/chenoweth/", x, y, "chenoweth", 100)
    x, y = build_matrix("training_set/baldwin/", x, y, "baldwin", 100)
    x, y = build_matrix("training_set/hader/", x, y, "hader", 100)
    x, y = build_matrix("training_set/carell/", x, y, "carell", 100)
    
    x = np.reshape(x, (600, 1025))
    y = np.reshape(y, (600, 6))
    
    x = x.T
    y = y.T
    
    # compute theta
    theta = gradient_descent2(f2, df2, x, y, init_theta, alpha)
    # print theta
    global theta_new_way
    theta_new_way = theta.T
    
    y_hat = np.dot(theta.T, x)
    y_hat = y_hat.T
    y = y.T
    
    for i in range(0, 600):
        row_max_index =  np.argmax(y_hat[i])
        if (y[i][row_max_index] == 1):
            success += 1
    
    success_rate = success/float(600)
    print "The performance on the training set: ", success_rate
    
    # build matrix for x2 and y2, using helper function for the validation set
    x2, y2 = build_matrix("validation_set/drescher/", x2, y2, "drescher", 10)
    x2, y2 = build_matrix("validation_set/ferrera/", x2, y2, "ferrera", 10)
    x2, y2 = build_matrix("validation_set/chenoweth/", x2, y2, "chenoweth", 10)
    x2, y2 = build_matrix("validation_set/baldwin/", x2, y2, "baldwin", 10)
    x2, y2 = build_matrix("validation_set/hader/", x2, y2, "hader", 10)
    x2, y2 = build_matrix("validation_set/carell/", x2, y2, "carell", 10)
    
    x2 = np.reshape(x2, (60, 1025))
    y2 = np.reshape(y2, (60, 6))
    
    y_hat2 = np.dot(theta.T, x2.T)
    y_hat2 = y_hat2.T
    
    for i in range(0, 60):
        row_max_index = np.argmax(y_hat2[i])
        if (y2[i][row_max_index] == 1):
            success2 += 1
            
    success_rate2 = success2/float(60)
    print "The performance on the validation set: ", success_rate2
    
    # call part 6 (d)
    print "\n"
    print "Part 6(d): Print out the results for Part 6(d)"
    test_gradient(x, y.T)
    
# part 8
# must run part 7 first in order to  assign value to the global variable 
# theta_new_way
def view_theta_new_way():
    for i in range(0, 6):
        theta = np.reshape(theta_new_way[i][1:], (32, 32))
        fig = figure(i)
        ax = fig.gca()
        graph = ax.imshow(theta, cmap=cm.coolwarm)
        fig.colorbar(graph)
        title("Heatmap of Row " + str(i) + " of Theta for Full Training Set")
        show()

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
    
# my main function
if __name__ == "__main__":
    # part 1
    print "--------------------------------------------------------------------"
    print "Part 1: Download images for actors"
    process_actor_image("facescrub_actors.txt")
    print "--------------------------------------------------------------------"
    print "\n"
    
    print "Part 1: Download images for actresses"
    process_actor_image("facescrub_actresses.txt")
    print "--------------------------------------------------------------------"
    print "\n"
    
    # part2
    print "Part 2: Separate dataset into three non-overlapping sets"
    separate_dataset()
    print "--------------------------------------------------------------------"
    print "\n"
    
    # part 3
    print "Part 3: Print out the results for part 3"
    test_dataset_for_hader_carell()
    print "--------------------------------------------------------------------"
    print "\n"
    
    # part 4
    print "Part 4: View the theta on full training set"
    view_full_theta()
    print "--------------------------------------------------------------------"
    print "\n"
    
    print "Part 4: View the theta on 2 images of each actor"
    view_partial_theta()
    print "--------------------------------------------------------------------"
    print "\n"
    
    # part 5
    print "Part 5: Test overfitting"
    test_overfit()
    print "--------------------------------------------------------------------"
    print "\n"
    
    print "Part 5: Test the performance of classifier on actors not including in act"
    print "Part 5: Download images for new actors"
    process_act_test_image("facescrub_actors.txt")
    print "\n"
    print "Part 5: Download images for new actresses"
    process_act_test_image("facescrub_actresses.txt")
    test_act_test()
    print "--------------------------------------------------------------------"
    print "\n"
    
    # part 6(d) and 7
    print "Part 6(d): Test gradient"
    print "Use theta obtained from Part 7 to test gradient, then run Part 7 first"
    print "Part 7: Print out the results for Part 7"
    test_dataset_new_way()
    print "--------------------------------------------------------------------"
    print "\n"
    
    # part 8
    print "Part 8: View thetas obtained from Part 7"
    view_theta_new_way()
    print "--------------------------------------------------------------------"
    
    
    
    
    
    
    
    
    
