import numpy as np
import gudhi as gh
import pandas as pd
import time
import os
# import matplotlib.pyplot as plt

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
center_of_mass = lambda diag: np.sum(diag, axis=0) / len(diag)
center_of_persistence = lambda diag: np.sum(
    np.array([diag[ii] * (diag[ii][1] - diag[ii][0]) for ii in range(len(diag))]), axis=0) / (
                                             np.sum(diag, axis=0)[1] - np.sum(diag, axis=0)[0])
center_of_inv_persistence = lambda diag: np.sum(
    np.array([diag[ii] / (diag[ii][1] - diag[ii][0]) for ii in range(len(diag))]), axis=0) / (
                                             np.sum(1 / (diag[:, 1] - diag[:, 0])))

tic = time.perf_counter()
# --------------------------------------------- PREPROCESSING ---------------------------------------------------
# Upload the data
subjects_dict = {"Human": [1, 30308], "Airplane": [61, 12892], "Ant": [81, 16772], "Bird": [241, 12946],
                 "FourLeg": [381, 20868], "Octopus": [121, 2682], "Fish": [221, 12148]}
category = "Bird"

df_category = pd.DataFrame(columns=["centroid", "persistence diagrams", "label"])
# SHOW IMAGE
# for i in np.arange(5):  # we take the first 5 elements for every kind of category
#     j = i + subjects_dict[category][0]
#     j = 241
#     f_off = open(fr'{cwd}/MeshsegBenchmark/data/off/{j}.off')
#     list_off = f_off.readlines()
#     f_off.close()
#     f_seg = open(fr'{cwd}/MeshsegBenchmark/data/seg/Benchmark/{j}/{j}_0.seg')
#     list_seg = f_seg.readlines()
#     f_seg.close()
#     list_number_seg = [int(a[:-1]) for a in list_seg]
#     l_index = 0
#     print("number of labels:", len(set(list_number_seg)))
#     centroids_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
#     for element_ind in np.arange(1, len(list_off)):
#         temp_number_list = np.array([float(e) for e in list_off[element_ind][:-1].split(" ")])
#         list_off[element_ind] = temp_number_list
#         if len(temp_number_list) == 4:
#             for c_ind in np.arange(len(set(list_number_seg))):
#                 if list_number_seg[l_index] == c_ind:
#                     centroids_dict[c_ind] += [(list_off[2 + int(temp_number_list[1])] + list_off[2 + int(temp_number_list[2])] + list_off[2 + int(temp_number_list[3])]) / 3]
#             l_index += 1
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# for p_ind in np.arange(len(set(list_number_seg))):
#     m = "o"
#     colors = list(matplotlib.colors.TABLEAU_COLORS.keys()) + ["lime", "pink"]
#     ax.scatter(np.array(centroids_dict[p_ind])[:, 0], np.array(centroids_dict[p_ind])[:, 1], np.array(centroids_dict[p_ind])[:, 2], marker=m, c=colors[p_ind])
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
# ax.grid(False)
# ax.set_axis_off()
# plt.show()

# # BUILD PERSISTENCE DIAGRAMS (AND SAVE THEM)
# for i in np.arange(5):  # we take the first 5 elements for every kind of category
#     print("starting", i+1, "out of 5 in", time.perf_counter()-tic)
#     j = i + subjects_dict[category]
#     f_off = open(fr'{cwd}/MeshsegBenchmark/data/off/{j}.off')
#     list_off = f_off.readlines()
#     f_off.close()
#     f_seg = open(fr'{cwd}/MeshsegBenchmark/data/seg/Benchmark/{j}/{j}_0.seg')
#     list_seg = f_seg.readlines()
#     f_seg.close()
#     list_number_seg = [int(a[:-1]) for a in list_seg]
#     faces_list = []
#     centroids_list = []
#
#     for element_ind in np.arange(1, len(list_off)):
#         temp_number_list = np.array([float(e) for e in list_off[element_ind][:-1].split(" ")])
#         list_off[element_ind] = temp_number_list
#         if len(temp_number_list) == 4:
#             faces_list += [[int(temp_number_list[1]), int(temp_number_list[2]), int(temp_number_list[3])]]
#             centroids_list += [(list_off[2+int(temp_number_list[1])]+list_off[2+int(temp_number_list[2])]+list_off[2+int(temp_number_list[3])])/3]
#
#     points_list = np.array(list_off[2:int(list_off[1][0]) + 2])
#     faces_list = np.array(faces_list)
#     # now in arr_points_from_off there are the centroids of the faces
#     # in list_number_seg there are the value of the segmentation associated with the faces
#     # now we want to compute for every centroid the 1-dim persistence homology of the cloud of points based on the geodetic distance from the chose point
#
#     # Initialise the solver to compute the geodesic distance
#     solver = pp3d.MeshHeatMethodDistanceSolver(points_list, faces_list)
#
#     for face_index in np.arange(int(list_off[1][1])):
#         face_geodesic_based_simplex_tree = gh.SimplexTree()
#         # Get the distance and the path between the source and target points
#
#         dist_from_selected_face = solver.compute_distance_multisource(faces_list[face_index])
#
#         times_for_faces = []
#         for face in faces_list:
#             times_for_faces += [max(dist_from_selected_face[face[0]], dist_from_selected_face[face[1]], dist_from_selected_face[face[2]])]
#         index_face_time_ordered_list = np.lexsort([times_for_faces])
#
#         for face_ind in index_face_time_ordered_list:
#             face_geodesic_based_simplex_tree.insert(faces_list[face_ind], times_for_faces[face_ind])
#
#         face_geodesic_based_persistence_diagrams = face_geodesic_based_simplex_tree.persistence()
#         face_label = list_number_seg[face_index]
#
#         df_category = df_category.append(pd.Series([centroids_list[face_index], face_geodesic_based_persistence_diagrams, face_label], index=df_category.columns), ignore_index=True)
#
# print(df_category)
# print(time.perf_counter()-tic)
# df_category.to_csv(fr'{cwd}/shape_segmentation/{category}.csv')

# LOAD OF CSV FILE
main = pd.read_csv(fr'{cwd}/shape_segmentation/{category}.csv')

# STORE THE PERSISTENCE DIAGRAMS AS NUMPY FILES (ONLY IF GENERATE THE PD MANUALLY)
# for index in main.index:
#     # restore the pd
#     p_d = main["persistence diagrams"][index].strip('][').split('), (')
#     p_d_list = []
#     for i in np.arange(len(p_d)):
#         if d == int(p_d[i].strip("()")[0]):
#             p_d_list += [[float(b) for b in p_d[i][4:].strip(")(").split(", ")]]
#     p_d = np.array(p_d_list)
#     if vsk_flag:
#         # add center of mass
#         p_d_1 = np.concatenate((p_d, [center_of_persistence(p_d)]), axis=0)
#         np.save(fr'{cwd}/shape_segmentation/{category}/vsk_pd_{index}.npy', p_d_1)
#     else:
#         np.load(fr'{cwd}/shape_segmentation/{category}/pd_{index}.npy')
#         # p_d = np.where(p_d > 6, 6, p_d)  # since all images are in the box [-1, 1]^3, we consider essential holes as holes with persistence 6 (the geodetic distance of two opposite points in the box)
#         p_d = np.array([c_p for c_p in p_d if c_p[1] < 6])  # remove essential holes
#         np.save(fr'{cwd}/shape_segmentation/{category}/pd_{index}.npy', p_d)

# LOAD PERSISTENCE DIAGRAMS
new_column_pd = []
for i in np.arange(len(main.index)):
    if vsk_flag:
        p_d = np.load(fr'{cwd}/shape_segmentation/{category}/vsk_pd_{i}.npy')  # imported the vsk pd
    else:
        p_d = np.load(fr'{cwd}/shape_segmentation/{category}/pd_{i}.npy')  # imported the pd

    new_column_pd += [p_d]

main['persistence diagram points'] = new_column_pd

toc_mid = time.perf_counter()
print(f"\ntotal time after data loading: {toc_mid - tic:f} seconds")

# ----------------------------------------- END OF PREPROCESSING ---------------------------------------------

n_fold = 5  # fold for cross validation
X = main['persistence diagram points']
y = main['label']

threshold_points, acceptable_size_main = train_test_split(y.index, test_size=0.01, random_state=42, stratify=y)
main_threshold = main['label'][threshold_points]

label_length, label_index = len(set(y)), list(acceptable_size_main)
for label_ in range(label_length):
    np.random.seed(1)
    label_index += list(np.random.choice(main_threshold.index[main_threshold == label_], 10))

X = main['persistence diagram points'][label_index]
y = main['label'][label_index]
print("points in main:", len(main))

report = {'t_train': [], 't_val': [], 'f1_score': [], 'accuracy': []}
for rand_state in program:
    persistence_kernel = 'PSWK'  # chosen kernel  PSWK - PWGK - PSSK

    balanced_train_index, balanced_test_index = train_test_split(y.index, test_size=0.3, random_state=rand_state,
                                                                 stratify=y)

    X_balanced_train = X.loc[balanced_train_index]
    y_balanced_train = y.loc[balanced_train_index]
    X_balanced_test = X.loc[balanced_test_index]
    y_balanced_test = y.loc[balanced_test_index]

    toc_mid2 = time.perf_counter()
    print(f"\ntime for splitting train and test data: {toc_mid2 - toc_mid:f} seconds")
    print("category: ", category)
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
                X_train, X_test = X_balanced_train[X_balanced_train.index[train_index]], X_balanced_train[
                    X_balanced_train.index[test_index]]
                y_train, y_test = y_balanced_train[y_balanced_train.index[train_index]], y_balanced_train[
                    y_balanced_train.index[test_index]]
                clf = SVC(kernel=PSSK, C=C, cache_size=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                f1s = f1_score(y_test, y_pred, average='macro')
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
            H_norm2 = w_F.T @ KFF @ w_F + w_G.T @ KGG @ w_G - 2 * w_F.T @ KFG @ w_G
            return np.exp(- H_norm2 / (2 * _tau ** 2))[0][0]


        def PWGK(XF, XG):  # XF and XG are array of persistence diagrams
            global Cp_w, rho, tau
            return np.array(
                [[persistance_weighted_gaussian_kernel(D1, D2, _Cp_w=Cp_w, _rho=rho, _tau=tau) for D2 in XG] for D1 in
                 XF])


        pers = lambda x: x[1] - x[0]
        # Training

        C_values = [0.1, 1, 10, 100, 1000]
        tau_values = [0.01, 0.1, 1, 10]
        rho_values = [1]  # [0.1, 1, 10, 100]
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
                X_train, X_test = X_balanced_train[X_balanced_train.index[train_index]], X_balanced_train[
                    X_balanced_train.index[test_index]]
                y_train, y_test = y_balanced_train[y_balanced_train.index[train_index]], y_balanced_train[
                    y_balanced_train.index[test_index]]
                clf = SVC(kernel=PWGK, C=C, cache_size=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                f1s = f1_score(y_test, y_pred, average='macro')
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
            theta = -np.pi / 2
            s = np.pi / _M
            # evaluating SW approximated routine
            for j in range(_M):
                v1 = np.dot(F, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v2 = np.dot(G, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v1_sorted = np.sort(v1, axis=0, kind='mergesort')
                v2_sorted = np.sort(v2, axis=0, kind='mergesort')
                SW += np.linalg.norm(v1_sorted - v2_sorted, 1) / _M
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
            theta = -np.pi / 2
            s = np.pi / _M
            # evaluating SW approximated routine
            for j in range(_M):
                v1 = np.dot(F, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v2 = np.dot(G, np.array([[np.cos(theta)], [np.sin(theta)]]))
                v1_sorted = np.sort(v1, axis=0, kind='mergesort')
                v2_sorted = np.sort(v2, axis=0, kind='mergesort')
                SW += np.linalg.norm(v1_sorted - v2_sorted, 1) / _M
                theta += s
            # now have SW(F,G)
            return SW


        def PSWM(XF, XG, _M):  # XF and XG are array of persistence diagrams
            return np.array(
                [[persistance_sliced_wasserstein_approximated_matrix(D1, D2, _M=_M) for D2 in XG] for D1 in XF])


        def PSWK(XF, XG):  # XF and XG are array of persistence diagrams
            global M, eta
            return np.array(
                [[persistance_sliced_wasserstein_approximated_kernel(D1, D2, _M=M, _eta=eta) for D2 in XG] for D1 in
                 XF])

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
    report['f1_score'] = report['f1_score'] + [f1_score(y_balanced_test, y_pred, average='macro')]
    report['accuracy'] = report['accuracy'] + [accuracy_score(y_balanced_test, y_pred)]

    # print(report)
print('-----------------------------------------------------------------------------------')
if vsk_flag:
    print('========== report of Sliced Wasserstein Kernel with VSPK in dimension %s ===========' % d)
else:
    print('========= report of Sliced Wasserstein Kernel without VSPK in dimension %s =========' % d)
print('-----------------------------------------------------------------------------------')
print("category: ", category)
print('time of training (mean out of %s):' % len(program), np.mean(report['t_train']))
print('time of validation (mean out of %s):' % len(program), np.mean(report['t_val']))
print('f1-score (mean out of %s):' % len(program), np.mean(report['f1_score']))
print('accuracy (mean out of %s):' % len(program), np.mean(report['accuracy']))
print('accuracy (std out of %s):' % len(program), np.std(report['accuracy']))
