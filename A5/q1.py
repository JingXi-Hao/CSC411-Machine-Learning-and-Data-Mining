'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import math

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    # loop over each digit in the range of 0-9
    for digit in range(10):
        # extract digits in the train_data which match the query label, the
        # shape for each digit_data is (700, 64)
        digit_data = data.get_digits_by_label(train_data, train_labels, digit)
        
        digit_means = digit_data.mean(0)
        means[digit, ] = digit_means
    
    # print(means)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for digit in range(10):
        digit_data = data.get_digits_by_label(train_data, train_labels, digit)
        
        # compute means for each digit
        digit_means = digit_data.mean(0).reshape((1, 64))
        
        # get total number of data for each digit
        num_of_data = digit_data.shape[0]
        
        cov_for_digit = np.zeros((64, 64))
        # loop over each train data obtained for each digit
        for index in range(num_of_data):
            x = digit_data[index, ].reshape((1, 64))
            diff = x - digit_means
            cov_for_digit = cov_for_digit + np.dot(diff.T, diff)
        
        # add 0.01I to ensure stability
        for_stability = np.identity(64) * 0.01
        covariances[digit, : , : ] = (cov_for_digit / num_of_data) + for_stability
    
    # print(covariances)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    n = digits.shape[0]
    log_likelihoods = np.zeros((n, 10))
    
    d = 64
    term1 = (-d/2) * math.log(2 * math.pi)
    
    # digits here is the train_data or test_data
    for i in range(n):
        data = digits[i, : ]
        for digit in range(10):
            digit_means = means[digit, : ].reshape((1, 64))
            digit_covs = covariances[digit, : , : ]
            
            covs_inverse = np.linalg.inv(digit_covs)
            diff = data - digit_means
            term3 = (-1/2) * (np.dot(np.dot(diff, covs_inverse), diff.T))
            
            term2 = (-1/2) * math.log(np.linalg.det(digit_covs))
            
            log_likelihoods[i][digit] = term1 +  term2 + term3[0][0]
    
    # print(log_likelihoods.shape)
    # print(log_likelihoods)
    return log_likelihoods

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    result_likelihoods = generative_likelihood(digits, means, covariances)
    
    n = digits.shape[0]
    
    results = np.zeros((n, 10))
    
    for i in range(n):
        data_likelihoods = result_likelihoods[i, : ]
        sum = np.sum(np.exp(data_likelihoods))
        
        for digit in range(10):
            computed_result = data_likelihoods[digit]  + math.log(1/10) - math.log(sum / 10)
            
            results[i][digit] = computed_result
    
    # print(results.shape)
    # print(results)
    return results

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    num_data = cond_likelihood.shape[0]
    
    result = 0
    
    for i in range(num_data):
        label = int(labels[i])
        result += cond_likelihood[i][label]
        
    result = result / num_data
    
    # Compute as described above and return
    # print (result)
    return result

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    # Compute and return the most likely class
    result = np.argmax(cond_likelihood, axis=1)
    return result
    
def compute_accuracy(labels, true_labels):
    n = labels.shape[0]
    count = 0
    
    for i in range(n):
        if int(labels[i]) == int(true_labels[i]):
            count += 1
        
    return count / n
    
def plot_leading_eigenvectors(covariances):
    for digit in range(10):
        covs = covariances[digit, : ]
        # find all eigenvectors and eigenvalues
        eigen_values, eigen_vectors = np.linalg.eig(covs)
        # find the index for the maximum eigenvalues
        index = np.argmax(eigen_values)
        eigen_vector = eigen_vectors[ : , index].reshape((8, 8))
        # plot the graphs side by side and make a 2 times 5 plot
        plt.subplot(2, 5, digit + 1)
        # show image in grayscale
        plt.imshow(eigen_vector, cmap='gray')
    plt.show()

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # question 1(a)
    train_avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    
    print('The average conditional log-likelihood on train set is {}.\n'.format(train_avg))
    
    print('The average conditional log-likelihood on test set is {}.\n'.format(test_avg))
    
    # question 1(b)
    classify_train = classify_data(train_data, means, covariances)
    classify_test = classify_data(test_data, means, covariances)
    
    train_acc = compute_accuracy(classify_train, train_labels)
    test_acc = compute_accuracy(classify_test, test_labels)
    
    print('The accuracy on train set is {}.\n'.format(train_acc))
    
    print('The accuracy on test set is {}.\n'.format(test_acc))
    
    # question 1(c)
    plot_leading_eigenvectors(covariances)

if __name__ == '__main__':
    main()
