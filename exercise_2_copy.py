import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import scipy

from scipy import ndimage as ndi
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage import data

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas
import h5py

import warnings
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

from skimage.feature import greycomatrix, greycoprops
from skimage import data

# --------------------
# tunable-parameters
# --------------------
train_path = 'Training set/'
test_path = 'Testing set/'
h5_data = 'Output/multi_data.h5'
h5_labels = 'Output/multi_labels.h5'
scoring = 'accuracy'
fixed_size = (256, 256)
bins = 8
seed = 9
split = 10
number_of_colors = 10
num_trees = 100


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


def color_shade(image, length, width):
    modified_image = image.reshape(length * width, 3)
    clf = KMeans(n_clusters=number_of_colors)
    KMeans_labels = clf.fit_predict(modified_image)
    counts = Counter(KMeans_labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    return rgb_colors


def gabor(image):
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    shrink = (slice(0, None, 3), slice(0, None, 3))
    image = img_as_float(image)

    # prepare reference features
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    ref_feats = feats
    return ref_feats


# https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv2-for-newbies-3c59796bf774
def kaze(image, vector_size=32):
    try:
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        if dsc is None:
            dsc = np.zeros(2048)
        else:
            dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None
    return dsc


def extract_image():
    image_path = 'Training set/bridge/gsun_0ad875050d4171cdfe1d6414f6a31415.jpg'
    image = cv2.imread(image_path)
    image = cv2.resize(image, fixed_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Hu moment
    feature = cv2.HuMoments(cv2.moments(gray_image))
    haralick = fd_haralick(image)
    # Haralick
    # select some patches from sky areas of the image
    PATCH_SIZE = 21
    sky_locations = [(54, 48), (21, 133), (90, 180), (195, 200)]
    sky_patches = []
    for loc in sky_locations:
        sky_patches.append(gray_image[loc[0]:loc[0] + PATCH_SIZE,
                           loc[1]:loc[1] + PATCH_SIZE])

    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (sky_patches):
        glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in sky_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[len(sky_patches):], ys[len(sky_patches):], 'bo',
            label='Sky')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(sky_patches):
        ax = fig.add_subplot(3, len(sky_patches), len(sky_patches) * 2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Loc %d' % (i + 1))

    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    # Gabor
    kernel = gabor_kernel(0.1, theta=45)
    gray_img = img_as_float(gray_image)
    # Normalize images for better comparison.
    img = (gray_img - gray_img.mean()) / gray_img.std()
    power = np.sqrt(ndi.convolve(img, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(img, np.imag(kernel), mode='wrap') ** 2)
    plt.imshow(power)
    plt.show()

    # Kaze
    alg = cv2.KAZE_create()
    # Dinding image keypoints
    kps = alg.detect(image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:32]
    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)
    kaze_image = cv2.drawKeypoints(gray_image, kps, outImage=None, color=(255, 0, 0))
    cv2.imwrite('kaze.jpg', kaze_image)

    # Histogram
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

    # Rgb color
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.show()

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def extract_feature():
    res_path = 'Feature set/'
    # get the training labels
    label_names = os.listdir(train_path)
    # sort the training labels
    label_names.sort()
    # empty lists to hold feature vectors and labels
    base_features = []
    color_features = []
    edge_texture_features = []
    invariant_features = []
    all_features = []
    wo_color_features = []
    wo_edge_texture_features = []
    wo_invariant_features = []

    labels = []
    for label_id, label in enumerate(label_names):
        if not label.startswith('.'):
            cur_train_path = os.path.join(train_path, label)
            cur_res_path = os.path.join(res_path, label)
            if not os.path.isdir(cur_res_path):
                os.mkdir(cur_res_path)
            image_names = os.listdir(cur_train_path)
            for img_id, image_name in enumerate(image_names):
                if not image_name.startswith('.'):
                    image_path = os.path.join(cur_train_path, image_name)
                    res_image_path = os.path.join(cur_res_path, image_name)
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, fixed_size)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    length = image.shape[0]
                    width = image.shape[1]
                    feature_length = length * width
                    ####################################
                    # Image extraction
                    ####################################
                    edge_image = cv2.Canny(gray_image, 100, 200)

                    des, kp = orb_feature(gray_image)
                    orb_image = cv2.drawKeypoints(gray_image, kp, outImage=None, color=(255, 0, 0))

                    cv2.imwrite(res_image_path[:-4] + '_gray.jpg', gray_image)
                    cv2.imwrite(res_image_path[:-4] + '_edge.jpg', edge_image)
                    cv2.imwrite(res_image_path[:-4] + '_orb.jpg', orb_image)

                    ####################################
                    # Global Feature extraction
                    ####################################
                    fv_hu_moments = fd_hu_moments(image)
                    fv_haralick = fd_haralick(image)
                    fv_histogram = fd_histogram(image)

                    rgb_colors = color_shade(image, length, width)
                    rgb_colors = [item for sublist in rgb_colors for item in sublist]
                    rgb_colors = np.array(rgb_colors)

                    gray_features = np.reshape(gray_image, feature_length)

                    edge_features = np.reshape(edge_image, feature_length)
                    gabor_features = gabor(gray_image).flatten()

                    orb_features = des.flatten()
                    orb_features = orb_features[:14400]
                    if len(orb_features) < 14400:
                        orb_features = np.pad(orb_features, (0, 14400-len(orb_features)), 'constant')

                    kaze_features = kaze(image)
                    ###################################
                    # Concatenate global features
                    ###################################
                    base_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
                    color_feature = np.hstack([base_feature, rgb_colors, gray_features])
                    edge_texture_feature = np.hstack([base_feature, edge_features, gabor_features])
                    invariant_feature = np.hstack([base_feature, orb_features, kaze_features])

                    all_feature = np.hstack([color_feature, edge_features, gabor_features, orb_features, kaze_features])
                    wo_color_feature = np.hstack(
                        [edge_texture_feature, orb_features, kaze_features])
                    wo_edge_texture_feature = np.hstack(
                        [color_feature, orb_features, kaze_features])
                    wo_invariant_feature = np.hstack(
                        [color_feature, edge_features, gabor_features])

                    # update the list of labels and feature vectors
                    labels.append(label)
                    base_features.append(base_feature)
                    color_features.append(color_feature)
                    edge_texture_features.append(edge_texture_feature)
                    invariant_features.append(invariant_feature)
                    all_features.append(all_feature)
                    wo_color_features.append(wo_color_feature)
                    wo_edge_texture_features.append(wo_edge_texture_feature)
                    wo_invariant_features.append(wo_invariant_feature)

    # get the overall feature vector size
    # print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

    # get the overall training label size
    # print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    le = LabelEncoder()
    target = le.fit_transform(labels)

    # scale features in the range (0-1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaled_features = scaler.fit_transform(global_features)

    # save the feature vector using HDF5
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(base_features))
    h5f_data.create_dataset('dataset_2', data=np.array(color_features))
    h5f_data.create_dataset('dataset_3', data=np.array(edge_texture_features))
    h5f_data.create_dataset('dataset_4', data=np.array(invariant_features))
    h5f_data.create_dataset('dataset_5', data=np.array(all_features))
    h5f_data.create_dataset('dataset_6', data=np.array(wo_color_features))
    h5f_data.create_dataset('dataset_7', data=np.array(wo_edge_texture_features))
    h5f_data.create_dataset('dataset_8', data=np.array(wo_invariant_features))
    h5f_data.close()
    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    h5f_label.close()


def orb_feature(gray_image):
    orb = cv2.ORB_create()
    kp = orb.detect(gray_image, None)
    kp, des = orb.compute(gray_image, kp)
    return des, kp


def classify():
    warnings.filterwarnings('ignore')
    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('linear_SVC', SVC(kernel='linear', random_state=seed)))
    models.append(('poly_SVC', SVC(kernel='poly', random_state=seed)))
    models.append(('rbf_SVC', SVC(kernel='rbf', random_state=seed)))
    models.append(('sigmoid_SVC', SVC(kernel='sigmoid', random_state=seed)))

    # variables to hold the results and names
    results = []
    names = []

    # import the feature vector and trained labels
    h5f_data = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')
    base_features_string = h5f_data['dataset_1']
    color_features_string = h5f_data['dataset_2']
    edge_texture_features_string = h5f_data['dataset_3']
    invariant_features_string = h5f_data['dataset_4']
    all_features_string = h5f_data['dataset_5']
    wo_color_features_string = h5f_data['dataset_6']
    wo_edge_texture_features_string = h5f_data['dataset_7']
    wo_invariant_features_string = h5f_data['dataset_8']

    global_labels_string = h5f_label['dataset_1']
    base_features = np.array(base_features_string)
    color_features = np.array(color_features_string)
    edge_texture_features = np.array(edge_texture_features_string)
    invariant_features = np.array(invariant_features_string)
    all_features = np.array(all_features_string)
    wo_color_features = np.array(wo_color_features_string)
    wo_edge_textures = np.array(wo_edge_texture_features_string)
    wo_invariant_features = np.array(wo_invariant_features_string)

    global_labels = np.array(global_labels_string)
    h5f_data.close()
    h5f_label.close()

    # verify the shape of the feature vector and labels
    # print("[STATUS] features shape: {}".format(global_features.shape))
    # print("[STATUS] labels shape: {}".format(global_labels.shape))
    # print("[STATUS] training started...")

    global_base_features = np.array(base_features)
    global_color_features = np.array(color_features)
    global_edge_texture_features = np.array(edge_texture_features)
    global_invariant_features = np.array(invariant_features)
    global_all_features = np.array(all_features)
    global_wo_color_features = np.array(wo_color_features)
    global_wo_edge_textures = np.array(wo_edge_textures)
    global_wo_invariant_features = np.array(wo_invariant_features)

    trainDataGlobals = []
    trainDataGlobals.append(global_base_features)
    # trainDataGlobals.append(global_color_features)
    trainDataGlobals.append(global_edge_texture_features)
    # trainDataGlobals.append(global_invariant_features)
    trainDataGlobals.append(global_all_features)
    # trainDataGlobals.append(global_wo_color_features)
    # trainDataGlobals.append(global_wo_edge_textures)
    # trainDataGlobals.append(global_wo_invariant_features)

    trainLabelsGlobal = np.array(global_labels)

    # 10-fold cross validation
    # for name, model in models:
    #     # for trainDataGlobal in trainDataGlobals:
    #     kfold = KFold(n_splits=split, random_state=seed)
    #     cv_results = cross_val_score(model, global_all_features, trainLabelsGlobal, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    #
    # # boxplot algorithm comparison
    # fig = plt.figure()
    # fig.suptitle('Machine Learning algorithm comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.show()

    test(global_all_features, global_base_features, global_color_features, global_edge_texture_features,
         global_invariant_features, global_wo_color_features, global_wo_edge_textures, global_wo_invariant_features,
         trainLabelsGlobal)


def test(global_all_features, global_base_features, global_color_features, global_edge_texture_features,
         global_invariant_features, global_wo_color_features, global_wo_edge_textures, global_wo_invariant_features,
         trainLabelsGlobal):
    # -----------------------------------
    # TESTING OUR MODEL
    # -----------------------------------
    # # create the model - SVM
    # base_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # base_clf.fit(global_base_features, trainLabelsGlobal)
    # color_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # color_clf.fit(global_color_features, trainLabelsGlobal)
    # edge_texture_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # edge_texture_clf.fit(global_edge_texture_features, trainLabelsGlobal)
    # invariant_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # invariant_clf.fit(global_invariant_features, trainLabelsGlobal)
    all_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # fit the training data to the model
    all_clf.fit(global_all_features, trainLabelsGlobal)
    # wo_color_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # wo_color_clf.fit(global_wo_color_features, trainLabelsGlobal)
    # wo_edge_texture_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # wo_edge_texture_clf.fit(global_wo_edge_textures, trainLabelsGlobal)
    # wo_invariant_clf = SVC(kernel='linear', C=1.0, random_state=seed)
    # # fit the training data to the model
    # wo_invariant_clf.fit(global_wo_invariant_features, trainLabelsGlobal)
    label_names = os.listdir(test_path)
    # get the training labels
    train_labels = os.listdir(train_path)
    # sort the training labels
    train_labels.sort()
    train_labels.remove('.DS_Store')
    train_labels = ['bridge', 'coast', 'mountain', 'rainforest']
    label_names = ['new_bridge', 'new_coast', 'new_mountain', 'new_rainforest']
    for label_id, label in enumerate(label_names):
        if not label.startswith('.'):
            cur_test_path = os.path.join(test_path, label)
            image_names = os.listdir(cur_test_path)
            # base_correct_count = 0
            # color_correct_count = 0
            # edge_texture_correct_count = 0
            # invariant_correct_count = 0
            all_correct_count = 0
            # wo_color_correct_count = 0
            # wo_edge_texture_correct_count = 0
            # wo_invariant_correct_count = 0
            total_count = 0
            for img_id, image_name in enumerate(image_names):
                if not image_name.startswith('.'):
                    # read the image
                    image_path = os.path.join(cur_test_path, image_name)
                    image = cv2.imread(image_path)
                    # resize the image
                    image = cv2.resize(image, fixed_size)

                    ####################################
                    # Global Feature extraction
                    ####################################
                    length = image.shape[0]
                    width = image.shape[1]
                    feature_length = length * width
                    fv_hu_moments = fd_hu_moments(image)
                    fv_haralick = fd_haralick(image)
                    fv_histogram = fd_histogram(image)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    edge_image = cv2.Canny(gray_image, 100, 200)
                    des, kp = orb_feature(gray_image)

                    rgb_colors = color_shade(image, length, width)
                    rgb_colors = [item for sublist in rgb_colors for item in sublist]
                    rgb_colors = np.array(rgb_colors)

                    gray_features = np.reshape(gray_image, feature_length)

                    edge_features = np.reshape(edge_image, feature_length)
                    gabor_features = gabor(gray_image).flatten()

                    orb_features = des.flatten()
                    orb_features = orb_features[:14400]
                    if len(orb_features) < 14400:
                        orb_features = np.pad(orb_features, (0, 14400 - len(orb_features)), 'constant')

                    kaze_features = kaze(image)
                    ###################################
                    # Concatenate global features
                    ###################################
                    base_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
                    color_feature = np.hstack([base_feature, rgb_colors, gray_features])
                    # edge_texture_feature = np.hstack([base_feature, edge_features, gabor_features])
                    # invariant_feature = np.hstack([base_feature, orb_features, kaze_features])

                    all_feature = np.hstack([color_feature, edge_features, gabor_features, orb_features, kaze_features])
                    # wo_color_feature = np.hstack(
                    #     [edge_texture_feature, orb_features, kaze_features])
                    # wo_edge_texture_feature = np.hstack(
                    #     [color_feature, orb_features, kaze_features])
                    # wo_invariant_feature = np.hstack(
                    #     [color_feature, edge_features, gabor_features])
                    # scale features in the range (0-1)
                    # scaler = MinMaxScaler(feature_range=(0, 1))
                    # rescaled_feature = scaler.fit_transform(global_feature.reshape(-1, 1))

                    # predict label of test image
                    # base_prediction = base_clf.predict(base_feature.reshape(1, -1))[0]
                    # color_prediction = color_clf.predict(color_feature.reshape(1, -1))[0]
                    # edge_texture_prediction = edge_texture_clf.predict(edge_texture_feature.reshape(1, -1))[0]
                    # invariant_prediction = invariant_clf.predict(invariant_feature.reshape(1, -1))[0]
                    all_prediction = all_clf.predict(all_feature.reshape(1, -1))[0]
                    # wo_color_prediction = wo_color_clf.predict(wo_color_feature.reshape(1, -1))[0]
                    # wo_edge_texture_prediction = wo_edge_texture_clf.predict(wo_edge_texture_feature.reshape(1, -1))[0]
                    # wo_invariant_prediction = wo_invariant_clf.predict(wo_invariant_feature.reshape(1, -1))[0]

                    print(image_path + train_labels[all_prediction])

                    # if label == train_labels[base_prediction]:
                    #     base_correct_count += 1
                    # if label == train_labels[color_prediction]:
                    #     color_correct_count += 1
                    # if label == train_labels[edge_texture_prediction]:
                    #     edge_texture_correct_count += 1
                    # if label == train_labels[invariant_prediction]:
                    #     invariant_correct_count += 1
                    if label[4:] == train_labels[all_prediction]:
                        all_correct_count += 1
                    # if label == train_labels[wo_color_prediction]:
                    #     wo_color_correct_count += 1
                    # if label == train_labels[wo_edge_texture_prediction]:
                    #     wo_edge_texture_correct_count += 1
                    # if label == train_labels[wo_invariant_prediction]:
                    #     wo_invariant_correct_count += 1
                    total_count += 1
            # print("For label: " + label + "base accuracy is: %f\n" % (base_correct_count / total_count))
            # print("For label: " + label + "color accuracy is: %f\n" % (color_correct_count / total_count))
            # print("For label: " + label + "edge_texture accuracy is: %f\n" % (edge_texture_correct_count / total_count))
            # print("For label: " + label + "invariant accuracy is: %f\n" % (invariant_correct_count / total_count))
            print("For label: " + label + "all accuracy is: %f\n" % (all_correct_count / total_count))
            # print("For label: " + label + "wo_color accuracy is: %f\n" % (wo_color_correct_count / total_count))
            # print("For label: " + label + "wo_edge_texture accuracy is: %f\n" % (
            #         wo_edge_texture_correct_count / total_count))
            # print("For label: " + label + "wo_invariant accuracy is: %f\n" % (wo_invariant_correct_count / total_count))


if __name__ == '__main__':
    # extract_feature()
    # classify()
    extract_image()

