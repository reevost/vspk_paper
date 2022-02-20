import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import itertools
from matplotlib import collections as mc
import mpl_toolkits.mplot3d.art3d as art3d
import itertools
from ripser import ripser
from persim import plot_diagrams
# from ripser import Rips
# from persim import PersistenceImager
import time

flag = False
tic = time.perf_counter()


def find_subsets(s_, n_):  # find all subset of length n of the set s.
    return list(itertools.combinations(s_, n_))  # return the list of all sub set


np.random.seed(7)

c, a = 0.8, 0.4  # parameters of torus and sphere
pp = 600  # number of points used for representation
# Make data.

# x = (c + a*np.cos(v))*np.cos(u)
# y = (c + a*np.cos(v))*np.sin(u)
# z = a*np.sin(v)
angle_range_u = np.array([np.random.rand(pp)*2*np.pi]).T
angle_range_v = np.array([np.random.rand(pp)*2*np.pi]).T
angle_range_t = np.array([np.random.rand(pp)*2*np.pi]).T
radius_range_r = np.array([np.random.rand(pp)*2*c-c]).T

torus_params = np.concatenate((angle_range_u, angle_range_v), axis=1)
sphere_params = np.concatenate((radius_range_r, angle_range_t), axis=1)

# torus_params = np.array([np.array([u, v]) for u in np.linspace(0, 2*np.pi, pp) for v in np.linspace(0, 2*np.pi, pp)])
# sphere_params = np.array([np.array([u, v]) for u in np.linspace(-c, c, pp) for v in np.linspace(0, 2*np.pi, pp)])

torus = np.array([[(c + a*np.cos(v))*np.cos(u), (c + a*np.cos(v))*np.sin(u), a*np.sin(v)] for (u, v) in torus_params])
torus += 0.1*np.random.random(torus.shape)

sphere = np.array([[np.sqrt(c**2-r**2)*np.cos(t), np.sqrt(c**2-r**2)*np.sin(t), r] for (r, t) in sphere_params])
sphere += 0.1*np.random.random(sphere.shape)

diagrams_sphere = ripser(sphere, maxdim=2, thresh=3.0)['dgms']
diagrams_torus = ripser(torus, maxdim=2, thresh=3.0)['dgms']

toc_mid = time.perf_counter()
print("\ntime after ripser: %f seconds" % (toc_mid-tic))

# construction of persistence barcodes
pers_lines_torus = {"h0": [], "h1": [], "h2": []}
for h in pers_lines_torus.keys():
    high = 0.0
    j = int(h[1])
    for i in np.arange(0, len(diagrams_torus[j])):
        if np.isfinite(diagrams_torus[j][i][1]):
            pers_lines_torus[h] += [[np.array([diagrams_torus[j][i][0], high]), np.array([diagrams_torus[j][i][1], high])]]
        else:
            pers_lines_torus[h] += [[np.array([diagrams_torus[j][i][0], high]), np.array([2., high])]]
        high += 1/len(diagrams_torus[j])

lc_t0 = mc.LineCollection(pers_lines_torus["h0"], colors='k', linewidths=1)
lc_t1 = mc.LineCollection(pers_lines_torus["h1"], colors='k', linewidths=1)
lc_t2 = mc.LineCollection(pers_lines_torus["h2"], colors='k', linewidths=1)

pers_lines_sphere = {"h0": [], "h1": [], "h2": []}
for h in pers_lines_sphere.keys():
    high = 0.0
    j = int(h[1])
    for i in np.arange(0, len(diagrams_sphere[j])):
        if np.isfinite(diagrams_sphere[j][i][1]):
            pers_lines_sphere[h] += [[np.array([diagrams_sphere[j][i][0], high]), np.array([diagrams_sphere[j][i][1], high])]]
        else:
            pers_lines_sphere[h] += [[np.array([diagrams_sphere[j][i][0], high]), np.array([2., high])]]
        high += 1/len(diagrams_sphere[j])

lc_s0 = mc.LineCollection(pers_lines_sphere["h0"], colors='k', linewidths=1)
lc_s1 = mc.LineCollection(pers_lines_sphere["h1"], colors='k', linewidths=1)
lc_s2 = mc.LineCollection(pers_lines_sphere["h2"], colors='k', linewidths=1)


toc_mid2 = time.perf_counter()
print("\ntime after barcodes: %f seconds" % (toc_mid2 - toc_mid))

if flag:
    # construction of vietoris rips complexes
    sphere_lines = []
    torus_lines = []
    sphere_triangles = []
    torus_triangles = []
    sphere_tetrahedrons = []
    torus_tetrahedrons = []
    # eps = 0.3  # resp. (sphere,torus) H2: 0.34 0.1438   -- H1: 0.2  0.21
    eps_s = 0.2
    eps_t = 0.5

    # vr-sphere construction
    # 1-dim case (lines)
    i = 0
    couple_list = find_subsets(sphere, 2)
    for couple in couple_list:
        if np.linalg.norm(couple[0] - couple[1]) < 2 * eps_s:
            sphere_lines += [[couple[0], couple[1]]]
    # 2-dim case  (triangles)
    triplet_list = find_subsets(sphere, 3)
    for triplet in triplet_list:
        couple_list_3 = find_subsets(triplet, 2)
        is_rips = True
        for couple_3 in couple_list_3:
            if np.linalg.norm(couple_3[0] - couple_3[1]) > 2 * eps_s:
                is_rips = False
        if is_rips:
            triangle = np.array([triplet[0], triplet[1], triplet[2]])
            sphere_triangles.append(triangle)
    # 3-dim case (tetrahedrons)
    quartet_list = find_subsets(sphere, 4)
    for quartet in quartet_list:
        couple_list_4 = find_subsets(quartet, 2)
        is_rips = True
        for couple_4 in couple_list_4:
            if np.linalg.norm(couple_4[0] - couple_4[1]) > 2 * eps_s:
                is_rips = False
        if is_rips:
            triplet_list_4 = find_subsets(quartet, 3)
            for triplet_4 in triplet_list_4:
                tetra_triangle = np.array([triplet_4[0], triplet_4[1], triplet_4[2]])
                sphere_tetrahedrons.append(tetra_triangle)

    toc_mid3 = time.perf_counter()
    print("\ntime after vr-sphere: %f seconds" % (toc_mid3 - toc_mid2))

    # vr-torus construction
    # 1-dim case (lines)
    couple_list = find_subsets(torus, 2)
    for couple in couple_list:
        if np.linalg.norm(couple[0] - couple[1]) < 2 * eps_t:
            torus_lines += [[couple[0], couple[1]]]
    # 2-dim case  (triangles)
    triplet_list = find_subsets(torus, 3)
    for triplet in triplet_list:
        couple_list_3 = find_subsets(triplet, 2)
        is_rips = True
        for couple_3 in couple_list_3:
            if np.linalg.norm(couple_3[0] - couple_3[1]) > 2 * eps_t:
                is_rips = False
        if is_rips:
            triangle = np.array([triplet[0], triplet[1], triplet[2]])
            torus_triangles.append(triangle)
    # 3-dim case (tetrahedrons)
    quartet_list = find_subsets(torus, 4)
    for quartet in quartet_list:
        couple_list_4 = find_subsets(quartet, 2)
        is_rips = True
        for couple_4 in couple_list_4:
            if np.linalg.norm(couple_4[0] - couple_4[1]) > 2 * eps_t:
                is_rips = False
        if is_rips:
            triplet_list_4 = find_subsets(quartet, 3)
            for triplet_4 in triplet_list_4:
                tetra_triangle = np.array([triplet_4[0], triplet_4[1], triplet_4[2]])
                torus_tetrahedrons.append(tetra_triangle)

    toc_mid4 = time.perf_counter()
    print("\ntime after vr-torus: %f seconds" % (toc_mid4 - toc_mid3))

toc = time.perf_counter()
print("\nrunning time: %f seconds" % (toc-tic))

# Plot the points.
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2])
plt.title("scattered sphere")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
plt.axis("off")
plt.figure(2)
plot_diagrams(diagrams_sphere, show=True)
plt.show()

fig = plt.figure(3)
ax = fig.gca()
ax.add_collection(lc_s0)
# ax.add_collection(mc.LineCollection([[np.array([0.2, 0]), np.array([0.2, 1])]], colors='r', linewidths=1))
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(-0.1, 1.1)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('persistence')
plt.title("sphere persistence barcodes - h0")
plt.show()
fig = plt.figure(4)
ax = fig.gca()
ax.add_collection(lc_s1)
# ax.add_collection(mc.LineCollection([[np.array([0.2, 0]), np.array([0.2, 1])]], colors='r', linewidths=1))
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(-0.1, 1.1)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('persistence')
plt.title("sphere persistence barcodes - h1")
plt.show()
fig = plt.figure(5)
ax = fig.gca()
ax.add_collection(lc_s2)
# ax.add_collection(mc.LineCollection([[np.array([0.2, 0]), np.array([0.2, 1])]], colors='r', linewidths=1))
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(-0.1, 1.1)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('persistence')
plt.title("sphere persistence barcodes - h2")
plt.show()

if flag:  # vr sphere
    fig = plt.figure(101)
    ax = fig.add_subplot(projection='3d')
    s_lc = art3d.Line3DCollection(sphere_lines, colors='k', linewidths=1)
    s_tri = art3d.Poly3DCollection(sphere_triangles, facecolors='b', alpha=0.1)
    s_tetra = art3d.Poly3DCollection(sphere_tetrahedrons, facecolors='r', alpha=0.1)
    ax.add_collection3d(s_tetra)
    ax.add_collection3d(s_tri)
    ax.add_collection3d(s_lc)
    ax.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    plt.axis("off")

    plt.title("scattered sphere, vietros-rips complex with eps = "+str(eps_s))
    plt.show()

fig = plt.figure(11)
ax = fig.add_subplot(projection='3d')
ax.scatter(torus[:, 0], torus[:, 1], torus[:, 2])
plt.title("scattered torus")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
plt.axis("off")
plt.figure(12)
plot_diagrams(diagrams_torus, show=True)
plt.show()

fig = plt.figure(13)
ax = fig.gca()
ax.add_collection(lc_t0)
ax.add_collection(mc.LineCollection([[np.array([0.5, 0]), np.array([0.5, 1])]], colors='r', linewidths=1))
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(-0.1, 1.1)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('persistence')
plt.title("torus persistence barcodes - h0")
plt.show()
fig = plt.figure(14)
ax = fig.gca()
ax.add_collection(lc_t1)
ax.add_collection(mc.LineCollection([[np.array([0.5, 0]), np.array([0.5, 1])]], colors='r', linewidths=1))
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(-0.1, 1.1)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('persistence')
plt.title("torus persistence barcodes - h1")
plt.show()
fig = plt.figure(15)
ax = fig.gca()
ax.add_collection(lc_t2)
ax.add_collection(mc.LineCollection([[np.array([0.5, 0]), np.array([0.5, 1])]], colors='r', linewidths=1))
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(-0.1, 1.1)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('persistence')
plt.title("torus persistence barcodes - h2")
plt.show()

if flag:  # vr torus
    fig = plt.figure(103)
    ax = fig.add_subplot(projection='3d')
    t_lc = art3d.Line3DCollection(torus_lines, colors='k', linewidths=1)
    t_tri = art3d.Poly3DCollection(torus_triangles, facecolors='b', alpha=0.1)
    t_tetra = art3d.Poly3DCollection(torus_tetrahedrons, facecolors='r', alpha=0.1)
    ax.add_collection3d(t_tetra)
    ax.add_collection3d(t_tri)
    ax.add_collection3d(t_lc)
    ax.scatter(torus[:, 0], torus[:, 1], torus[:, 2])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    plt.axis("off")

    plt.title("scattered torus, vietros-rips complex with eps = "+str(eps_t))
    plt.show()


