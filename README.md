### Code repository of the preprint: 
# Variably Scaled Persistence Kernels (VSPK) for persistence homology applications.

#### Abstract

In recent years, various kernels have been proposed in the context of per- sistent homology to deal with persistence diagrams in supervised learning ap- proaches. In this paper, we consider the idea of variably scaled kernels, for ap- proximating functions and data, and we interpret them in the framework of per- sistent homology. We call them Variably Scaled Persistence Kernels (VSPKs). These new kernels are then tested in different classification experiments. The obtained results show that they can improve the performance and the efficiency of existing standard kernels.
Keywords: kernel-based learning; variably scaled persistence kernel; persistence diagrams; persistent homology


#### Setup and Requirements

First unzip the diagrams.zip file, which contains all the persistence digrams built for the application of VSPK in the Alzheimer's disease diagnosis from the Oasis3 dataset [^5].
All the diagrams here are built with Ripser [^4]. 

Code runs with python 3.8.x. The libraries needed are:

- numpy 1.21.4
- gudhi 3.5
- pandas 1.3.4
- scipy 1.7.2
- scikit-learn 1.0.1 
- potpourri3d 0.0.6

#### Launch

To run the code enter the following command

python filename.py

where filename can be chosen between 

- SW_oasis
- PWG_oasis
- PSS_oasis

To run the code relative to the results on the Alzheimer's disease diagnosis, respectively handled with the Sliced Wasserstein (SW) kernel [^1], the Persistence Weighted Gaussian (PWG) kernel [^2] and the Persistence Scale-Space (PSS) kernel [^3].

- discrete_dynamical_model_analysis

To run the code related to the application in orbit recognition.

- shape_segmentation_analysis

To run the code related to the shape segmentation task.



[^1]: M. Carriere, M. Cuturi, S. Oudot, Sliced wasserstein kernel for persistence diagrams, in: International Conference on Machine Learning, PMLR, 2017, pp. 664–673.
[^2]: G. Kusano, K. Fukumizu, Y. Hiraoka, Kernel method for persistence diagrams via kernel embedding and weight factor, The Journal of Machine Learning Research 18 (1) (2017) 6947–6987.
[^3]: J. Reininghaus, S. Huber, U. Bauer, R. Kwitt, A stable multi-scale kernel for topological machine learning, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 4741–4748.
[^4]: C. Tralie, N. Saul, R. Bar-On, Ripser.py: A lean persistent homology li- brary for python, The Journal of Open Source Software 3 (29) (2018) 925. doi:10.21105/joss.00925.
[^5]: P. J. LaMontagne, T. L. Benzinger, J. C. Morris, S. Keefe, R. Hornbeck, C. Xiong, E. Grant, J. Hassenstab, K. Moulder, A. Vlassenko, et al., Oasis- 3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease, MedRxiv (2019).
