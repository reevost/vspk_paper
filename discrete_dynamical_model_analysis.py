import numpy as np
import gudhi as gh
import pandas as pd
import time
import os

from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy.spatial import distance_matrix
from sklearn.model_selection import KFold

vsk_flag = True # for Variably Scaled Persistence kernel version
# vsk_flag = False # for original kernel version
np.random.seed(42)
program = np.arange(10)
d = 1  # dimension of the feature
cwd = os.getcwd()  # get the working directory

# define the possible psi for variable scaled persistence kernel framework
center_of_mass = lambda diag: np.sum(diag, axis=0)/len(diag)
center_of_persistence = lambda diag: np.sum(np.array([diag[ii]*(diag[ii][1]-diag[ii][0]) for ii in range(len(diag))]), axis=0) / (np.sum(diag, axis=0)[1]-np.sum(diag, axis=0)[0])
center_of_inv_persistence = lambda diag: np.sum(np.array([diag[ii]/(diag[ii][1]-diag[ii][0]) for ii in range(len(diag))]), axis=0) / (np.sum(1/(diag[:, 1]-diag[:, 0])))


tic = time.perf_counter()
# --------------------------------------------- PREPROCESSING ---------------------------------------------------
coefficient_r_list = [2.5, 3.5, 4.0, 4.1, 4.3]

# Generate the dynamical system
# np.random.seed(7)
# starting_points = np.random.random((100, 2))
# dynamic_df = pd.DataFrame(columns=["starting point", "dynamic points"])
# r_ind = 4  # parameter of the system
# r = coefficient_r_list[r_ind]
#
# for i in np.arange(5):
#     x_0_y_0 = starting_points[i]
#     dynamic_point_list = [x_0_y_0]
#     for n in np.arange(1000):
#         x_n_plus_1 = (dynamic_point_list[-1][0] + r * dynamic_point_list[-1][1] * (1 - dynamic_point_list[-1][1])) % 1
#         # y_n_plus_1 = (dynamic_point_list[-1][1] + r * dynamic_point_list[-1][0] * (1 - dynamic_point_list[-1][0])) % 1
#         y_n_plus_1 = (dynamic_point_list[-1][1] + r * x_n_plus_1 * (1 - x_n_plus_1)) % 1
#         dynamic_point_list += [[x_n_plus_1, y_n_plus_1]]
#     dynamic_df.loc[i] = [x_0_y_0, np.array(dynamic_point_list)]
#
# for i in np.arange(5):
#     fig = plt.figure(i+1)
#     ax = fig.add_subplot()
#     ax.scatter(dynamic_df["dynamic points"][i][:, 0], dynamic_df["dynamic points"][i][:, 1])
#     plt.show()
#
#
# # BUILD PERSISTENCE DIAGRAMS (AND SAVE THEM)
# dict_sub_dict = {}
#
# for i in np.arange(100):
#     toc = time.perf_counter()
#
#     dynamic_data_array = dynamic_df["dynamic points"][i]
#     rips_complex = gh.RipsComplex(points=dynamic_data_array, max_edge_length=1)
#     rips_structure_tree = rips_complex.create_simplex_tree(max_dimension=2)
#     persistence_diagrams = rips_structure_tree.persistence(min_persistence=-1)
#     print("the time needed to compute iteration %d is %f" % (i, time.perf_counter()-toc))
#     if plot_flag:
#         ax = gh.plot_persistence_diagram(persistence=persistence_diagrams, legend=True)
#         ax.set_title("Persistence of case %d" % i)
#         ax.set_aspect("equal")  # forces to be square shaped
#         plt.show()
#     persistence_diagrams_savable = []
#     for dp in persistence_diagrams:
#         persistence_diagrams_savable += [[dp[0], dp[1][0], dp[1][1]]]
#     print("len of pd", len(persistence_diagrams_savable))
#     np.save(r'%s/dynamic_diagrams_2/%s_d%s.npy' % (cwd, r_ind, i), np.array(persistence_diagrams_savable))

# LOAD OF PERSISTENCE DIAGRAMS
main = pd.DataFrame(columns=['persistence diagram points', 'y'])

new_column_pd = []
new_column_y = []
for jnd in np.arange(len(coefficient_r_list)):
    for i in np.arange(50):
        p_d = np.load(r'%s/dynamic_diagrams/%s_d%s.npy' % (cwd, jnd, i))  # imported the full pd
        p_d_1 = p_d[np.nonzero(p_d[:, 0])][:, 1:]  # selecting only the 1-dim features
        p_d_10 = p_d_1[:10, :]  # take only the 10 feature with higher persistent

        if vsk_flag:
            # add center of mass
            p_d = np.concatenate((p_d_10, [center_of_persistence(p_d_10)]), axis=0)  # Psi_a
            # p_d = np.concatenate((p_d_10, [center_of_persistence(p_d_1[10:, :])]), axis=0)  # Psi_rho
        new_column_pd += [p_d]
        new_column_y += [jnd]

main['persistence diagram points'] = new_column_pd
main['y'] = new_column_y

toc_mid = time.perf_counter()
print(f"\ntotal time after diagrams evaluation and data preprocessing: {toc_mid - tic:f} seconds")

# ----------------------------------------- END OF PREPROCESSING ---------------------------------------------

n_fold = 5  # fold for cross validation
#  Create train and test set for d1
main0 = main[main['y'] == 0]
main1 = main[main['y'] == 1]
main2 = main[main['y'] == 2]
main3 = main[main['y'] == 3]
main4 = main[main['y'] == 4]

X = main['persistence diagram points']
y = main['y']
X0 = main0['persistence diagram points']
y0 = main0['y']
X1 = main1['persistence diagram points']
y1 = main1['y']
X2 = main2['persistence diagram points']
y2 = main2['y']
X3 = main3['persistence diagram points']
y3 = main3['y']
X4 = main4['persistence diagram points']
y4 = main4['y']

report = {'t_train': [], 't_val': [], 'f1_score': [], 'accuracy': []}
for rand_state in program:
    persistence_kernel = 'PSWK'  # chosen kernel  PSWK - PWGK - PSSK

    train_index0, test_index0 = train_test_split(y0.index, test_size=0.3, random_state=rand_state)
    train_index1, test_index1 = train_test_split(y1.index, test_size=0.3, random_state=rand_state)
    train_index2, test_index2 = train_test_split(y2.index, test_size=0.3, random_state=rand_state)
    train_index3, test_index3 = train_test_split(y3.index, test_size=0.3, random_state=rand_state)
    train_index4, test_index4 = train_test_split(y4.index, test_size=0.3, random_state=rand_state)

    balanced_train_index = np.concatenate((train_index0, train_index1, train_index2, train_index3, train_index4), axis=0)
    balanced_test_index = np.concatenate((test_index0, test_index1, test_index2, test_index3, test_index4), axis=0)

    X_balanced_train = X.loc[balanced_train_index]
    y_balanced_train = y.loc[balanced_train_index]
    X_balanced_test = X.loc[balanced_test_index]
    y_balanced_test = y.loc[balanced_test_index]

    toc_mid2 = time.perf_counter()
    print(f"\ntime for splitting train and test data: {toc_mid2 - toc_mid:f} seconds")

    print('-----------------------------------------------------------------------------------')
    print('--------------------------------- new round ---------------------------------------')
    print('-----------------------------------------------------------------------------------')
    if vsk_flag:
        print('=============================== with VSK variant ==================================')
    toc_mid2 = time.perf_counter()
    if persistence_kernel == 'PSSK':
        print('======================== Persistence Scale-Space Kernel ===========================')

        def persistance_scale_space_kernel(F, G, _sigma):  # F, G are arrays of the points of persistance diagrams
            # evaluate the kernel, supposing there is no eternal hole
            dist_matrix = distance_matrix(F, G)
            dist_matrix_bar = distance_matrix(F, G[:, ::-1])  # supposed G.shape = (*, 2)
            sum_matrix = np.exp(-dist_matrix**2/(8*_sigma))-np.exp(-dist_matrix_bar**2/(8*_sigma))
            return np.sum(sum_matrix)/(8*np.pi*_sigma)


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
                X_train, X_test = X_balanced_train[X_balanced_train.index[train_index]], X_balanced_train[X_balanced_train.index[test_index]]
                y_train, y_test = y_balanced_train[y_balanced_train.index[train_index]], y_balanced_train[y_balanced_train.index[test_index]]
                clf = SVC(kernel=PSSK, C=C, cache_size=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
                f1s = f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4], average='macro')
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
        print(f"\ntime for cross validation: {toc_mid3 - toc_mid2:f} seconds")

        sigma = best_sigma  # PSSK
        print('best sigma: ', sigma, '\nbest C: ', best_C)
        classifier = SVC(kernel=PSSK, C=best_C, cache_size=1000)
        classifier.fit(X_balanced_train, y_balanced_train)

        y_pred = classifier.predict(X_balanced_test)
        print(classification_report(y_balanced_test, y_pred))

        toc_mid4 = time.perf_counter()
        print(f"\ntime for classifying: {toc_mid4 - toc_mid3:f} seconds")


    elif persistence_kernel == 'PWGK':
        print('====================== Persistence Weighted Gaussian Kernel =======================')


        def persistance_weighted_gaussian_kernel(F, G, _Cp_w, _rho, _tau):
            # F, G are arrays of the points of persistance diagrams
            # Cp_w = (C, p) 2-tuple contains the parameter of p_arc
            # rho is the parameter of the gaussian kernel
            # tau is the parameter of the persistence gaussian kernel
            # evaluate the kernel, supposing there is no eternal hole
            w_arc = lambda x: np.arctan(_Cp_w[0] * (pers(x)) ** _Cp_w[1])
            w_F = np.array([[w_arc(x)] for x in F])
            w_G = np.array([[w_arc(z)] for z in G])
            KFG = np.exp(-distance_matrix(F, G) ** 2 / (2 * _rho ** 2))
            KFF = np.exp(-distance_matrix(F, F) ** 2 / (2 * _rho ** 2))
            KGG = np.exp(-distance_matrix(G, G) ** 2 / (2 * _rho ** 2))
            # ||E_kg(F)-E_kg(G)||_Hk
            H_norm2 = w_F.T@KFF@w_F + w_G.T@KGG@w_G - 2 * w_F.T@KFG@w_G
            return np.exp(- H_norm2 / (2 * _tau**2))[0][0]


        def PWGK(XF, XG):  # XF and XG are array of persistence diagrams
            global Cp_w, rho, tau
            return np.array([[persistance_weighted_gaussian_kernel(D1, D2, _Cp_w=Cp_w, _rho=rho, _tau=tau) for D2 in XG] for D1 in XF])


        pers = lambda x: x[1] - x[0]
        # Training

        C_values = [0.1, 1, 10, 100, 1000]
        tau_values = [0.01, 0.1, 1, 10]
        rho_values = [0.1, 1, 10, 100]
        p = 5  # [1, 5, 10] possible values for p, but to reduce cv time we fix to 5, for which we have stability
        C_w_values = [0.1, 1, 10, 100]
        PWGK_param = [(c, t, r, C_w) for c in C_values for t in tau_values for r in rho_values for C_w in C_w_values]
        best_mean, best_C, best_tau, best_rho, best_C_w = 0, 1, 1, 1, 1
        progress = 0
        for param in PWGK_param:
            # tac = time.perf_counter()
            C, tau, rho, C_w = param[0], param[1], param[2], param[3]
            Cp_w = (C_w, p)
            kf = KFold(n_splits=n_fold, shuffle=True, random_state=7)
            metric_output = []
            for train_index, test_index in kf.split(X_balanced_train):
                X_train, X_test = X_balanced_train[X_balanced_train.index[train_index]], X_balanced_train[X_balanced_train.index[test_index]]
                y_train, y_test = y_balanced_train[y_balanced_train.index[train_index]], y_balanced_train[y_balanced_train.index[test_index]]
                clf = SVC(kernel=PWGK, C=C, cache_size=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
                f1s = f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4], average='macro')
                # as loss function can be used also the rmse
                # root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
                # metric_output.append(root_mse)
                metric_output.append(f1s)
            mean = np.mean(metric_output)
            if mean > best_mean:
                best_mean = mean
                best_tau = tau
                best_rho = rho
                best_C_w = C_w
                best_C = C
            progress += 1
            print("progress %d/%d, in %f seconds" % (progress, len(PWGK_param), time.perf_counter() - toc_mid2))

        toc_mid3 = time.perf_counter()
        print(f"\ntime for cross validation: {toc_mid3 - toc_mid2:f} seconds")

        tau, rho, Cp_w = best_tau, best_rho, (best_C_w, p)
        print('best_C: ', best_C, '\nbest tau: ', tau, '\nbest rho: ', rho, '\nbest Cp_w: ', Cp_w)
        classifier = SVC(kernel=PWGK, C=best_C, cache_size=1000)
        classifier.fit(X_balanced_train, y_balanced_train)

        y_pred = classifier.predict(X_balanced_test)
        print(classification_report(y_balanced_test, y_pred))

        toc_mid4 = time.perf_counter()
        print(f"\ntime for classifying: {toc_mid4 - toc_mid3:f} seconds")



    elif persistence_kernel == 'PSWK':
        print('==================== Persistence Sliced Wasserstein Kernel ========================')


        def persistance_sliced_wasserstein_approximated_kernel(F, G, _M, _eta):
            # F, G are arrays of the points of persistance diagrams
            # eta is the coefficient of the associated gaussian kernel
            # M is the number of direction in the half circle. 6 is sufficient, 10 or more is like do not approximate
            # evaluate the kernel, supposing there is no eternal hole
            # for each persistence diagram project points in diagonal and add to the points associated with other persistence diagram
            eps = 1 / (2 * _eta ** 2)
            Diag_F = (F + F[:, ::-1]) / 2
            Diag_G = (G + G[:, ::-1]) / 2
            F = np.vstack((F, Diag_G))
            G = np.vstack((G, Diag_F))
            SW = 0
            theta = -np.pi/2
            s = np.pi/_M
            # evaluating SW approximated routine
            for j in range(_M):
                v1 = np.dot(F, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v2 = np.dot(G, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v1_sorted = np.sort(v1, axis=0, kind='mergesort')
                v2_sorted = np.sort(v2, axis=0, kind='mergesort')
                SW += np.linalg.norm(v1_sorted-v2_sorted, 1)/_M
                theta += s
            # now have SW(F,G)
            return np.exp(-eps * SW)  # eps = 1/(2*sigma**2)


        def persistance_sliced_wasserstein_approximated_matrix(F, G, _M):
            # F, G are arrays of the points of persistance diagrams
            # M is the number of direction in the half circle. 6 is sufficient, 10 or more is like do not approximate
            # evaluate the kernel, supposing there is no eternal hole
            # for each persistence diagram project points in diagonal and add to the points associated with other persistence diagram
            Diag_F = (F + F[:, ::-1]) / 2
            Diag_G = (G + G[:, ::-1]) / 2
            F = np.vstack((F, Diag_G))
            G = np.vstack((G, Diag_F))
            SW = 0
            theta = -np.pi/2
            s = np.pi/_M
            # evaluating SW approximated routine
            for j in range(_M):
                v1 = np.dot(F, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v2 = np.dot(G, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v1_sorted = np.sort(v1, axis=0, kind='mergesort')
                v2_sorted = np.sort(v2, axis=0, kind='mergesort')
                SW += np.linalg.norm(v1_sorted-v2_sorted, 1)/_M
                theta += s
            # now have SW(F,G)
            return SW


        def PSWM(XF, XG, _M):  # XF and XG are array of persistence diagrams
            return np.array([[persistance_sliced_wasserstein_approximated_matrix(D1, D2, _M=_M) for D2 in XG] for D1 in XF])


        def PSWK(XF, XG):  # XF and XG are array of persistence diagrams
            global M, eta
            return np.array([[persistance_sliced_wasserstein_approximated_kernel(D1, D2, _M=M, _eta=eta) for D2 in XG] for D1 in XF])
        

        # Training
        C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        eta_values = [0.01, 0.1, 1, 10, 100] * 3
        M = 10
        PSWK_param = [(c, e) for c in C_values for e in eta_values]
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=7)
        progress = 0
        metric_output = [0 for a in PSWK_param]
        for train_index, test_index in kf.split(X_balanced_train):
            X_train, X_test = X_balanced_train[X_balanced_train.index[train_index]], X_balanced_train[
                X_balanced_train.index[test_index]]
            y_train, y_test = y_balanced_train[y_balanced_train.index[train_index]], y_balanced_train[
                y_balanced_train.index[test_index]]

            # evaluation of the baseline of the kernel matrix
            SW_train = PSWM(X_train, X_train, M)
            SW_test_train = PSWM(X_test, X_train, M)

            flat = np.sort(np.matrix.flatten(SW_train), kind='mergesort')
            d1_n, d5_n, d9_n = (len(flat) + 1) / 10, (len(flat) + 1) / 2, 9 * (len(flat) + 1) / 10
            flat = np.concatenate(([0], flat))
            d1 = flat[int(d1_n)] + (d1_n - int(d1_n)) * (flat[int(d1_n) + 1] - flat[int(d1_n)])
            d5 = flat[int(d5_n)] + (d5_n - int(d5_n)) * (flat[int(d5_n) + 1] - flat[int(d5_n)])
            d9 = flat[int(d9_n)] + (d9_n - int(d9_n)) * (flat[int(d9_n) + 1] - flat[int(d9_n)])
            d1, d5, d9 = np.sqrt(d1), np.sqrt(d5), np.sqrt(d9)
            eta_values = [d1 * 0.01, d1 * 0.1, d1, d1 * 10, d1 * 100, d5 * 0.01, d5 * 0.1, d5, d5 * 10, d5 * 100,
                          d9 * 0.01, d9 * 0.1, d9, d9 * 10, d9 * 100]
            PSWK_param = [(c, e) for c in C_values for e in eta_values]
            for ind in range(len(PSWK_param)):
                C, eta = PSWK_param[ind][0], PSWK_param[ind][1]
                # using base kernel and parameters to quickly evaluate the kernel
                gram_SW_train = np.exp(-SW_train / (2 * eta ** 2))
                gram_SW_test_train = np.exp(-SW_test_train / (2 * eta ** 2))
                clf = SVC(kernel='precomputed', C=C, cache_size=1000)
                clf.fit(gram_SW_train, y_train)
                y_pred = clf.predict(gram_SW_test_train)
                cm = confusion_matrix(y_test, y_pred)
                f1s = f1_score(y_test, y_pred, average='macro')
                # as loss function can be used also the rmse
                # root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
                metric_output[ind] += f1s  # root_mse
            progress += 1
            print(f"progress {progress:d}/{n_fold:d}, in {time.perf_counter() - toc_mid2:f} seconds")

        best_mean = np.max(metric_output)
        best_mean_ind = np.where(metric_output == best_mean)[0][0]

        toc_mid3 = time.perf_counter()
        print(f"\ntime for cross validation: {toc_mid3 - toc_mid2:f} seconds")

        best_C = PSWK_param[best_mean_ind][0]
        eta = PSWK_param[best_mean_ind][1]

        print("best eta: ", eta, '\nbest C: ', best_C)
        classifier = SVC(kernel=PSWK, C=best_C, cache_size=1000)
        classifier.fit(X_balanced_train, y_balanced_train)

        y_pred = classifier.predict(X_balanced_test)
        print(classification_report(y_balanced_test, y_pred))

        toc_mid4 = time.perf_counter()
        print(f"\ntime for classifying: {toc_mid4 - toc_mid3:f} seconds")

    report['t_train'] = report['t_train'] + [toc_mid3 - toc_mid2]
    report['t_val'] = report['t_val'] + [toc_mid4 - toc_mid3]
    report['f1_score'] = report['f1_score'] + [f1_score(y_balanced_test, y_pred, labels=[0, 1, 2, 3, 4], average='macro')]
    report['accuracy'] = report['accuracy'] + [accuracy_score(y_balanced_test, y_pred)]

    # print(report)
print('-----------------------------------------------------------------------------------')
if vsk_flag:
    print('========== report of Sliced Wasserstein Kernel with VSPK in dimension %s ===========' % d)
else:
    print('========= report of Sliced Wasserstein Kernel without VSPK in dimension %s =========' % d)
print('-----------------------------------------------------------------------------------')
print('time of training (mean out of %s):' % len(program), np.mean(report['t_train']))
print('time of validation (mean out of %s):' % len(program), np.mean(report['t_val']))
print('f1-score (mean out of %s):' % len(program), np.mean(report['f1_score']))
print('accuracy (mean out of %s):' % len(program), np.mean(report['accuracy']))
print('accuracy (std out of %s):' % len(program), np.std(report['accuracy']))
