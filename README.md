#$\mu C^{1}$ and $\mu C^{2}$: Geometric Algebra-based models for gesture recognition and time-series classification
$\mu C^{1}$ and $\mu C^{2}$ are two algorithms based on geometric algebra and nearest-neighbor classification (and inspired by stroke gesture recognition) which were originally designed for recognizing Sign Language gestures. In its original Matlab implementation, these algorithms computed high accuracies and achieved fast training and classification times on a custom Sign Language dataset (see the paper referenced in the end of this section).
We are making our code (translated to Python) publicly available for reproducibility and so that researchers can also use these algorithms on their own time-series classification problems. We are also open to modifications to improve the code.

cliffordTS.py includes a READ ME section where it is explained how to use the CliffordClassifier. Moreover, example.py shows a concrete example on a publicly available multivariate time series dataset.

If you use the contents of this respository for your work, please cite the following paper:

•	A. Calado, P. Roselli, V. Errico, N. Magrofuoco, J. Vanderdonckt, and G. Saggio, “A Geometric Model-Based Approach to Hand Gesture Recognition”, IEEE Transactions on Systems, Man, and Cybernetics, vol. 52, no. 10, pp. 6151–6161, 2022.
