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

flag = False
vsk_flag = False
psi_flag = False
program = ['PSWK']
d = 1

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

# BUILD PERSISTENCE DIAGRAMS (AND SAVE THEM)
# dict_sub_dict = {}
#
# for sub in df_ctx.index:
#     # print('subject', sub)
#     toc_mid = time.perf_counter()
#
#     subject_4d_data_list = []
#     for label in df_3d.index:
#         # print('label', label)
#         subject_4d_data_list += [[df_3d.loc[label, 'x'], df_3d.loc[label, 'y'], df_3d.loc[label, 'z'], df_ctx.loc[sub, label]]]  # x, y, z, thickness
#
#     subject_4d_data_array = np.array(subject_4d_data_list)
#     diagrams = ripser(subject_4d_data_array, maxdim=3)['dgms']
#
#     print("\ntime for build diagram of subject %s: %f seconds" % (sub, time.perf_counter() - toc_mid))
#
#     dict_sub_dict[sub] = diagrams
#     for dim in range(4):
#         np.save(r'/Users/federicolot/PycharmProjects/Unipd/TESI/diagrams/%s_d%s.npy' % (sub, dim), diagrams[dim])

# graphics of where is the center of persistence in the diagram
# a_persistence_diagram_1 = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/diagrams/%s_d%s.npy' % ("OAS30345", 1))
# his_center_of_persistence_1 = np.concatenate((a_persistence_diagram_1, [center_of_persistence(a_persistence_diagram_1)]), axis=0)
# a_persistence_diagram_2 = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/diagrams/%s_d%s.npy' % ("OAS30345", 2))
# his_center_of_persistence_2 = np.concatenate((a_persistence_diagram_2, [center_of_persistence(a_persistence_diagram_2)]), axis=0)
#
# plt.figure(1)
# plot_diagrams(a_persistence_diagram_1, show=True, labels="H1")
# plt.show()
# plt.figure(2)
# plot_diagrams(his_center_of_persistence_1, show=True, labels="H1")
# plt.show()
# plt.figure(3)
# plot_diagrams(a_persistence_diagram_2, show=True, labels="H1")
# plt.show()
# plt.figure(4)
# plot_diagrams(his_center_of_persistence_2, show=True, labels="H1")
# plt.show()


# LOAD OF PERSISTENCE DIAGRAMS
main = df_y.filter('y')

for dim in range(1, 3):  # decide what dimension include
    new_column = []
    for subj in main.index:
        p_d = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/diagrams/%s_d%s.npy' % (subj, dim))
        if vsk_flag:
            # add center of mass
            p_d = np.concatenate((p_d, [center_of_persistence(p_d)]), axis=0)
        new_column += [p_d]
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

    X_cop_train, X_cop_test, y_cop_train, y_cop_test = train_test_split(X_cop, y_cop, test_size=0.3, random_state=42)

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

print(main)

for persistence_kernel in program:

    train_index0, test_index0 = train_test_split(y0.index, test_size=0.3, random_state=7)
    train_index1, test_index1 = train_test_split(y1.index, test_size=0.3, random_state=7)

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


    elif persistence_kernel == 'PWGK':
        print('====================== Persistence Weighted Gaussian Kernel =======================')


        # def persistance_weighted_linear_kernel(F, G, Cp_w, m_rff, rho):  # F, G are arrays of the points of persistance diagrams
        #     # evaluate the kernel, supposing there is no eternal hole
        #     w_arc = lambda x: np.arctan(Cp_w[0] * (pers(x)) ** Cp_w[1])
        #     # gaussian_matrix = np.exp(-eps*distance_matrix(F, G)**2)  ---> eps = 1/(2*sigma**2)  computable O(m**2*n**2) HIGH
        #     # using random fourier features
        #     N = 2  # len(F[0]) not need to compute really
        #     w_F_ = np.array([[w_arc(x)] for x in F])
        #     w_F = np.vstack((w_F_, w_F_))
        #     w_G_ = np.array([[w_arc(x)] for x in G])
        #     w_G = np.vstack((w_G_, w_G_))
        #     z = np.random.multivariate_normal(mean=np.zeros(N), cov=np.eye(N)/rho**2, size=(m_rff, ))
        #     # exp_B_F = np.exp(+1j * z @ F.T) @ w_F_  # can return a complex value
        #     # exp_B_G = np.exp(-1j * z @ G.T) @ w_G_
        #     B_F = np.hstack((np.cos(z @ F.T)/np.sqrt(len(F)), np.sin(z @ F.T)/np.sqrt(len(F)))) @ w_F
        #     B_G = np.hstack((np.cos(z @ G.T)/np.sqrt(len(G)), np.sin(z @ G.T)/np.sqrt(len(G)))) @ w_G
        #     # print("exp:", 1/m_rff*exp_B_F.T@exp_B_G)
        #     # print("cos:", 1/m_rff*B_F.T@B_G)
        #     return 1/m_rff*B_F.T@B_G  # or *sum(B_F*B_G)
        #     # don't compute the "internal" gaussian kernel but directly k(F,G)


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
        p = 10  # [1, 5, 10] possible values for p, but to reduce cv time we fix to 10, for which we have stability
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
                X_train, X_test = X_balanced_train[train_index], X_balanced_train[test_index]
                y_train, y_test = y_balanced_train[train_index], y_balanced_train[test_index]
                clf = SVC(kernel=PWGK, C=C, cache_size=1000)
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
                best_tau = tau
                best_rho = rho
                best_C_w = C_w
                best_C = C
            progress += 1
            print("progress %d/%d, in %f seconds" % (progress, len(PWGK_param), time.perf_counter() - toc_mid2))

        toc_mid3 = time.perf_counter()
        print("\ntime for cross validation: %f seconds" % (toc_mid3 - toc_mid2))

        tau, rho, Cp_w = best_tau, best_rho, (best_C_w, p)
        print('best_C: ', best_C, '\nbest tau: ', tau, '\nbest rho: ', rho, '\nbest Cp_w: ', Cp_w)
        classifier = SVC(kernel=PWGK, C=best_C, cache_size=1000)
        classifier.fit(X_balanced_train, y_balanced_train)

        y_pred = classifier.predict(X_balanced_test)
        print(classification_report(y_balanced_test, y_pred))

        toc_mid4 = time.perf_counter()
        print("\ntime for classifying: %f seconds" % (toc_mid4 - toc_mid3))



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
        eta_values = [0.01, 0.1, 1, 10, 100]*3
        M = 10
        PSWK_param = [(c, e) for c in C_values for e in eta_values]
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=7)
        progress = 0
        metric_output = [0 for a in PSWK_param]
        for train_index, test_index in kf.split(X_balanced_train):
            X_train, X_test = X_balanced_train[train_index], X_balanced_train[test_index]
            y_train, y_test = y_balanced_train[train_index], y_balanced_train[test_index]

            # evaluation of the baseline of the kernel matrix
            SW_train = PSWM(X_train, X_train, M)
            SW_test_train = PSWM(X_test, X_train, M)
            flat = np.sort(np.matrix.flatten(SW_train), kind='mergesort')
            d1_n, d5_n, d9_n = (len(flat)+1)/10, (len(flat)+1)/2, 9*(len(flat)+1)/10
            flat = np.concatenate(([0], flat))
            d1 = flat[int(d1_n)] + (d1_n - int(d1_n)) * (flat[int(d1_n)+1] - flat[int(d1_n)])
            d5 = flat[int(d5_n)] + (d5_n - int(d5_n)) * (flat[int(d5_n)+1] - flat[int(d5_n)])
            d9 = flat[int(d9_n)] + (d9_n - int(d9_n)) * (flat[int(d9_n)+1] - flat[int(d9_n)])
            d1, d5, d9 = np.sqrt(d1), np.sqrt(d5), np.sqrt(d9)
            eta_values = [d1*0.01, d1*0.1, d1, d1*10, d1*100, d5*0.01, d5*0.1, d5, d5*10, d5*100, d9*0.01, d9*0.1, d9, d9*10, d9*100]
            PSWK_param = [(c, e) for c in C_values for e in eta_values]
            for ind in range(len(PSWK_param)):
                C, eta = PSWK_param[ind][0], PSWK_param[ind][1]
                # using base kernel and parameters to quickly evaluate the kernel
                gram_SW_train = np.exp(-SW_train / (2 * eta**2))
                gram_SW_test_train = np.exp(-SW_test_train / (2 * eta ** 2))
                clf = SVC(kernel='precomputed', C=C, cache_size=1000)
                clf.fit(gram_SW_train, y_train)
                y_pred = clf.predict(gram_SW_test_train)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                f1s = f1_score(cm)
                # as loss function can be used also the rmse
                # root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
                metric_output[ind] += f1s  # root_mse
            progress += 1
            print("progress %d/%d, in %f seconds" % (progress, n_fold, time.perf_counter() - toc_mid2))

        best_mean = np.max(metric_output)
        best_mean_ind = np.where(metric_output == best_mean)[0][0]

        toc_mid3 = time.perf_counter()
        print("\ntime for cross validation: %f seconds" % (toc_mid3 - toc_mid2))

        best_C = PSWK_param[best_mean_ind][0]
        eta = PSWK_param[best_mean_ind][1]

        print('best eta: ', eta, '\nbest C: ', best_C)
        classifier = SVC(kernel=PSWK, C=best_C, cache_size=1000)
        classifier.fit(X_balanced_train, y_balanced_train)

        y_pred = classifier.predict(X_balanced_test)
        print(classification_report(y_balanced_test, y_pred))

        toc_mid4 = time.perf_counter()
        print("\ntime for classifying: %f seconds" % (toc_mid4 - toc_mid3))

    elif persistence_kernel == 'PFK':
        print('=========================== Persistence Fisher Kernel =============================')


        def persistance_fisher_kernel(F, G, _t, _delta):  # F, G are arrays of the points of persistance diagrams
            # evaluate the kernel, supposing there is no eternal hole
            # for each persistence diagram project points in diagonal and add to the points associated with other persistence diagram
            #  Variant with NO FGT
            Diag_F = (F + F[:, ::-1]) / 2
            Diag_G = (G + G[:, ::-1]) / 2
            new_F = np.vstack((F, Diag_G))
            new_G = np.vstack((G, Diag_F))
            Omega = np.vstack((new_F, new_G))
            # normalization (uniformly) - variant normalize with persistence
            w_scalar = 1. / len(new_F)
            # evaluating p_F
            temp_pi = np.zeros((len(Omega), len(new_F)))
            for ii in range(len(new_F)):
                temp_pi[:, ii] = multivariate_normal.pdf(Omega, mean=new_F[ii],
                                                         cov=_delta * np.eye(2))  # 2 = len(new_F[0]) so it will be always 2
            temp_pi *= w_scalar
            pi = np.sum(temp_pi, axis=1)

            # evaluating p_G
            temp_pi = np.zeros((len(Omega), len(new_G)))
            for jj in range(len(new_G)):
                temp_pi[:, jj] = multivariate_normal.pdf(Omega, mean=new_G[jj],
                                                         cov=_delta * np.eye(2))  # 2 = len(new_G[0]) so it will be always 2
            temp_pi *= w_scalar
            pj = np.sum(temp_pi, axis=1)

            # normalization
            pi /= sum(pi)
            pj /= sum(pj)

            # Hellinger mapping
            si = np.sqrt(pi)
            sj = np.sqrt(pj)
            dot = np.dot(si, sj)
            if dot < 1:
                dist_fim = np.arccos(dot)
                # print("dot product", dot)
            else:
                dist_fim = 0.0
                # print("dot product", dot)

            return np.exp(- dist_fim / _t)


        def persistance_fisher_matrix(F, G, _delta):
            # evaluate the kernel, supposing there is no eternal hole
            # for each persistence diagram project points in diagonal and add to the points associated with other persistence diagram
            #  Variant with NO FGT
            Diag_F = (F + F[:, ::-1]) / 2
            Diag_G = (G + G[:, ::-1]) / 2
            new_F = np.vstack((F, Diag_G))
            new_G = np.vstack((G, Diag_F))
            Omega = np.vstack((new_F, new_G))
            # normalization (uniformly) - variant normalize with persistence
            w_scalar = 1. / len(new_F)
            # evaluating p_F
            temp_pi = np.zeros((len(Omega), len(new_F)))
            for ii in range(len(new_F)):
                temp_pi[:, ii] = multivariate_normal.pdf(Omega, mean=new_F[ii],
                                                         cov=_delta * np.eye(2))  # 2 = len(new_F[0]) so it will be always 2
            temp_pi *= w_scalar
            pi = np.sum(temp_pi, axis=1)

            # evaluating p_G
            temp_pi = np.zeros((len(Omega), len(new_G)))
            for jj in range(len(new_G)):
                temp_pi[:, jj] = multivariate_normal.pdf(Omega, mean=new_G[jj],
                                                         cov=_delta * np.eye(2))  # 2 = len(new_G[0]) so it will be always 2
            temp_pi *= w_scalar
            pj = np.sum(temp_pi, axis=1)

            # normalization
            pi /= sum(pi)
            pj /= sum(pj)

            # Hellinger mapping
            si = np.sqrt(pi)
            sj = np.sqrt(pj)
            dot = np.dot(si, sj)
            if dot < 1:
                dist_fim = np.arccos(dot)
                # print("dot product", dot)
            else:
                dist_fim = 0.0
                # print("dot product", dot)

            return dist_fim


        def PFM(XF, XG, _delta):  # XF and XG are array of persistence diagrams
            return np.array([[persistance_fisher_matrix(D1, D2, _delta=_delta) for D2 in XG] for D1 in XF])


        def PFK(XF, XG):  # XF and XG are array of persistence diagrams
            global delta, t
            return np.array([[persistance_fisher_kernel(D1, D2, _delta=delta, _t=t) for D2 in XG] for D1 in XF])


        # Training


        C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # t_values = [1, 2, 5, 10, 20, 50]
        t_values = [1, 5, 20, 50]*3
        delta_values = [0.01, 0.1, 1, 10, 100]
        PFK_param = [(c, t) for c in C_values for t in t_values]
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=None)
        progress = 0
        metric_output = np.zeros((len(delta_values), len(PFK_param)))

        # optional - save the gram matrix
        # for delta_t in delta_values:
        #     print(delta_t, 'done')
        #     total_FM = PFM(X, X, _delta=delta_t)
        #     np.save(r'/Users/federicolot/PycharmProjects/Unipd/TESI/fisher_metric_matrix/%dfisher_matrix_%f.npy' % (d, delta_t), total_FM)

        global_train_index = []
        for subj in balanced_train_index:
            index = np.where(subj == X.index)[0][0]
            global_train_index += [index]
        global_train_index = np.array(global_train_index)

        global_test_index = []
        for subj in balanced_test_index:
            index = np.where(subj == X.index)[0][0]
            global_test_index += [index]
        global_test_index = np.array(global_test_index)

        for train_index, test_index in kf.split(X_balanced_train):
            local_train_train_index, local_train_test_index = global_train_index[train_index], global_train_index[test_index]
            y_train, y_test = y_balanced_train[train_index], y_balanced_train[test_index]

            for delta_ind in range(len(delta_values)):
                delta = delta_values[delta_ind]

                # evaluation of the baseline of the kernel matrix - GRAM MATRIX VARIANT
                # print('fisher_matrix_%.2f.npy' % delta)
                # total_delta_fisher_matrix = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/fisher_metric_matrix/%dfisher_matrix_%.2f.npy' % (d,delta))
                # print('zeroes in matrix: ', np.count_nonzero(total_delta_fisher_matrix == 0))
                # FM_train = total_delta_fisher_matrix[np.ix_(local_train_train_index, local_train_train_index)]
                # FM_test_train = total_delta_fisher_matrix[np.ix_(local_train_test_index, local_train_train_index)]

                X_train = X[local_train_train_index]
                X_test = X[local_train_test_index]
                FM_train = PFM(X_train, X_train, _delta=delta)
                FM_test_train = PFM(X_test, X_train, _delta=delta)
                flat = np.sort(np.matrix.flatten(FM_train), kind='mergesort')
                flat = flat[np.flatnonzero(flat)]
                flat = np.concatenate(([0], flat))
                temp_t_values = []
                for q in [1, 5, 20, 50]:
                    dq_n = q*(len(flat)) / 100
                    dq = flat[int(dq_n)] + (dq_n - int(dq_n)) * (flat[int(dq_n) + 1] - flat[int(dq_n)])
                    temp_t_values += [dq/1000]
                    temp_t_values += [dq]
                    temp_t_values += [dq*1000]

                t_values = temp_t_values
                PFK_param = [(c, t) for c in C_values for t in t_values]
                for ind in range(len(PFK_param)):
                    C, t = PFK_param[ind][0], PFK_param[ind][1]
                    gram_FM_train = np.exp(-FM_train / t)
                    gram_FM_test_train = np.exp(-FM_test_train / t)
                    clf = SVC(kernel='precomputed', C=C, cache_size=1000)
                    clf.fit(gram_FM_train, y_train)
                    y_pred = clf.predict(gram_FM_test_train)
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    f1s = f1_score(cm)
                    # as loss function can be used also the rmse
                    # root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    metric_output[delta_ind, ind] += f1s  # root_mse
                progress += 1
                print("progress %d/%d, in %f seconds" % (progress, n_fold*len(delta_values), time.perf_counter() - toc_mid2))


        best_mean = np.max(metric_output)
        best_mean_ind = np.where(metric_output == best_mean)

        toc_mid3 = time.perf_counter()
        print("\ntime for cross validation: %f seconds" % (toc_mid3 - toc_mid2))

        delta = delta_values[best_mean_ind[0][0]]
        best_C = PFK_param[best_mean_ind[1][0]][0]
        t = PFK_param[best_mean_ind[1][0]][1]

        print('best delta: ', delta, '\nbest t: ', t, '\nbest C: ', best_C)
        classifier = SVC(kernel=PFK, C=best_C, cache_size=1000)
        classifier.fit(X_balanced_train, y_balanced_train)

        y_pred = classifier.predict(X_balanced_test)
        print(classification_report(y_balanced_test, y_pred))

        # GRAM MATRIX VARIANT
        # total_delta_fisher_matrix = np.load(r'/Users/federicolot/PycharmProjects/Unipd/TESI/fisher_metric_matrix/%dfisher_matrix_%.2f.npy' % (d, delta))
        # FM_train = total_delta_fisher_matrix[np.ix_(global_train_index, global_train_index)]
        # FM_test_train = total_delta_fisher_matrix[np.ix_(global_test_index, global_train_index)]
        #
        # gram_FM_train = np.exp(-FM_train / t)
        # gram_FM_test_train = np.exp(-FM_test_train / t)
        # clf = SVC(kernel='precomputed', C=best_C, cache_size=1000)
        # clf.fit(gram_FM_train, y_balanced_train)
        # y_pred = clf.predict(gram_FM_test_train)
        # print(classification_report(y_balanced_test, y_pred))

        toc_mid4 = time.perf_counter()
        print("\ntime for classifying: %f seconds" % (toc_mid4 - toc_mid3))

