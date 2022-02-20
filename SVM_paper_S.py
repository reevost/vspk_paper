import numpy as np
import matplotlib.pyplot as plt
# from ripser import ripser
from persim import plot_diagrams
# from ripser import Rips
from persim import PersistenceImager
import pandas as pd
import time

from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance_matrix
from sklearn.model_selection import KFold

vsk_flag = True
psi_flag = False
np.random.seed(42)
program = np.arange(10)  # np.random.randint(1000, size=2)
d = 1  # dimension of the feature

# define statistic function for confusion matrix (in binary classification framework)
f1_score = lambda c_m:  2*c_m[1][1] / (2 * c_m[1][1] + c_m[0][1] + c_m[1][0])
precision = lambda c_m:  c_m[1][1] / (c_m[1][1] + c_m[0][1])
recall = lambda c_m:  c_m[1][1] / (c_m[1][1] + c_m[1][0])
accuracy = lambda c_m:  (c_m[0][0] + c_m[1][1]) / (c_m[0][0] + c_m[0][1] + c_m[1][0] + c_m[1][1])

# define the possible psi for variable scaled persistence kernel framework
center_of_mass = lambda diag: np.sum(diag, axis=0)/len(diag)
center_of_persistence = lambda diag: np.sum(np.array([diag[i]*(diag[i][1]-diag[i][0]) for i in range(len(diag))]), axis=0) / (np.sum(diag, axis=0)[1]-np.sum(diag, axis=0)[0])
center_of_inv_persistence = lambda diag: np.sum(np.array([diag[i]/(diag[i][1]-diag[i][0]) for i in range(len(diag))]), axis=0) / (np.sum(1/(diag[:, 1]-diag[:, 0])))


tic = time.perf_counter()
# --------------------------------------------- PREPROCESSING ---------------------------------------------------
# Read CSV file into DataFrame
df_y = pd.read_csv('oasis3_final_dataset.csv', index_col=0)

# LOAD OF PERSISTENCE DIAGRAMS
main = df_y.filter('y')


for dim in range(1, 3):  # decide what dimension include, in this case 1 and 2.
    new_column = []
    for subj in main.index:
        p_d = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/diagrams/%s_d%s.npy' % (subj, dim))
        p_d_10 = p_d[np.argsort(p_d[:, 1]-p_d[:, 0])[::-1][:10]]  # take only the 10 feature with higher persistent
        p_d_11 = p_d[np.argsort(p_d[:, 1]-p_d[:, 0])[::-1][10:]]
        if vsk_flag and dim == 1:
            # add center of mass
            # p_d = np.concatenate((p_d, [center_of_persistence(p_d)]), axis=0)
            p_d_10 = np.concatenate((p_d_10, [center_of_persistence(p_d_11)]), axis=0)
        new_column += [p_d_10]
    main['d%s' % dim] = new_column

# test the discriminative power of psi(D)
if psi_flag:
    cop = []
    for sub in main.index:
        pers_diag = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/diagrams/%s_d2.npy' % sub)
        cop += [center_of_persistence(pers_diag)]

    # ANALYSIS OF CENTER OF PERSISTENCE IMPACT ON CLASSIFICATION
    main_cop = main
    main_cop['c_o_p'] = cop
    cop = np.array(cop)
    main_cop = main_cop.drop(['d2', 'd1'], axis=1)
    print(main_cop)

    y_cop = main_cop['y']
    X_cop = cop

    X_cop_train, X_cop_test, y_cop_train, y_cop_test = train_test_split(X_cop, y_cop, test_size=0.3, random_state=7)

    C_range = np.logspace(-4, 7, 13)
    gamma_range = np.logspace(-8, 3, 13)
    param_grid = [(c, g) for c in C_range for g in gamma_range]
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    best_C, best_gamma, best_mean, q = 1, 1, 0, 0
    for param in param_grid:
        C, gamma = param[0], param[1]
        metric_output = []
        for train_index, test_index in kf.split(X_cop_train):
            X_train, X_test = X_cop_train[train_index], X_cop_train[test_index]
            y_train, y_test = y_cop_train[train_index], y_cop_train[test_index]
            clf = SVC(kernel='rbf', C=C, gamma=gamma, cache_size=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            f1s = f1_score(cm)
            # as loss function can be used also the rmse
            # root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
            # metric_output.append(root_mse)
            metric_output.append(f1s)
        mean = np.mean(metric_output)
        if mean > best_mean:
            best_mean = mean
            best_gamma = gamma
            best_C = C
        q += 1
        print('done %d/%d' % (q, len(param_grid)))
    print("The best parameters are C = %f, and gamma = %f" % (best_C, best_gamma))

    clf = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
    clf.fit(X_cop_train, y_cop_train)
    y_pred = clf.predict(X_cop_test)
    print(classification_report(y_cop_test, y_pred))

toc_mid = time.perf_counter()
print("\ntotal time after diagrams evaluation and data preprocessing: %f seconds" % (toc_mid - tic))

# ----------------------------------------- END OF PREPROCESSING ---------------------------------------------

n_fold = 5  # fold for cross validation
#  Create train and test set for d1 or d2
if d == 1:
    main = main.drop('d2', axis=1)
    main0 = main[main['y'] == 0]
    main1 = main[main['y'] == 1]

    X = main['d1']
    y = main['y']
    X1 = main1['d1']
    y1 = main1['y']
    X0 = main0['d1']
    y0 = main0['y']
else:
    main = main.drop('d1', axis=1)
    main0 = main[main['y'] == 0]
    main1 = main[main['y'] == 1]

    X = main['d2']
    y = main['y']
    X1 = main1['d2']
    y1 = main1['y']
    X0 = main0['d2']
    y0 = main0['y']

report = {'t_train': [], 't_val': [], 'f1_score': [], 'accuracy': []}
for rand_state in program:

    train_index0, test_index0 = train_test_split(y0.index, test_size=0.3, random_state=rand_state)
    train_index1, test_index1 = train_test_split(y1.index, test_size=0.3, random_state=rand_state)

    balanced_train_index = np.concatenate((train_index0, train_index1), axis=0)
    balanced_test_index = np.concatenate((test_index0, test_index1), axis=0)

    X_balanced_train = X.loc[balanced_train_index]
    y_balanced_train = y.loc[balanced_train_index]
    X_balanced_test = X.loc[balanced_test_index]
    y_balanced_test = y.loc[balanced_test_index]

    toc_mid2 = time.perf_counter()
    print("\ntime for splitting train and test data: %f seconds" % (toc_mid2 - toc_mid))

    print('-----------------------------------------------------------------------------------')
    print('--------------------------------- new round ---------------------------------------')
    print('-----------------------------------------------------------------------------------')
    if vsk_flag:
        print('=============================== with VSK variant ==================================')
    toc_mid2 = time.perf_counter()


    print('====================== Persistence Scale Space Kernel =============================')


    def persistance_scale_space_kernel(F, G, _sigma):  # F, G are arrays of the points of persistance diagrams
        # evaluate the kernel, supposing there is no eternal hole
        dist_matrix = distance_matrix(F, G)
        dist_matrix_bar = distance_matrix(F, G[:, ::-1])  # supposed G.shape = (*, 2)
        sum_matrix = np.exp(-dist_matrix ** 2 / (8 * _sigma)) - np.exp(-dist_matrix_bar ** 2 / (8 * _sigma))
        return np.sum(sum_matrix) / (8 * np.pi * _sigma)


    def PSSK(XF, XG):  # XF and XG are array of persistence diagrams
        global sigma  # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]. spectrum of possible values
        return np.array([[persistance_scale_space_kernel(D1, D2, _sigma=sigma) for D2 in XG] for D1 in XF])


    # Training

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    sigma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    PSSK_param = [(c, s) for c in C_values for s in sigma_values]
    best_mean, best_C, best_sigma = 0, 1, 1  # PSSK
    progress = 0
    for param in PSSK_param:
        # tac = time.perf_counter()
        C, sigma = param[0], param[1]
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=None)
        metric_output = []
        for train_index, test_index in kf.split(X_balanced_train):
            X_train, X_test = X_balanced_train[train_index], X_balanced_train[test_index]
            y_train, y_test = y_balanced_train[train_index], y_balanced_train[test_index]
            clf = SVC(kernel=PSSK, C=C, cache_size=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            f1s = f1_score(cm)
            # as loss function can be used also the rmse
            # root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
            # metric_output.append(root_mse)
            metric_output.append(f1s)
        mean = np.mean(metric_output)
        if mean > best_mean:
            best_mean = mean
            best_sigma = sigma
            best_C = C
        progress += 1
        print("progress %d/%d, in %f seconds" % (progress, len(PSSK_param), time.perf_counter() - toc_mid2))

    toc_mid3 = time.perf_counter()
    print("\ntime for cross validation: %f seconds" % (toc_mid3 - toc_mid2))

    sigma = best_sigma  # PSSK
    print('best sigma: ', sigma, '\nbest C: ', best_C)
    classifier = SVC(kernel=PSSK, C=best_C, cache_size=1000)
    classifier.fit(X_balanced_train, y_balanced_train)

    y_pred = classifier.predict(X_balanced_test)
    print(classification_report(y_balanced_test, y_pred))

    toc_mid4 = time.perf_counter()
    print("\ntime for classifying: %f seconds" % (toc_mid4 - toc_mid3))

    report['t_train'] = report['t_train'] + [toc_mid3 - toc_mid2]
    report['t_val'] = report['t_val'] + [toc_mid4 - toc_mid3]
    report['f1_score'] = report['f1_score'] + [f1_score(confusion_matrix(y_balanced_test, y_pred, labels=[0, 1]))]
    report['accuracy'] = report['accuracy'] + [accuracy(confusion_matrix(y_balanced_test, y_pred, labels=[0, 1]))]

# print(report)
print('-----------------------------------------------------------------------------------')
if vsk_flag:
    print('========== report of Persistence Scale Space Kernel with VSPK in dimension %s ===========' % d)
else:
    print('========= report of Persistence Scale Space Kernel without VSPK in dimension %s =========' % d)
print('-----------------------------------------------------------------------------------')
print('time of training (mean out of %s):' % len(program), np.mean(report['t_train']))
print('time of validation (mean out of %s):' % len(program), np.mean(report['t_val']))
print('f1-score (mean out of %s):' % len(program), np.mean(report['f1_score']))
print('accuracy (mean out of %s):' % len(program), np.mean(report['accuracy']))
print('accuracy (std out of %s):' % len(program), np.std(report['accuracy']))

