import os
from time import perf_counter
import click
import numpy as np
import torch
from sklearn.svm import SVC


#----------------------------------------------------------------------------

@click.command()
@click.option('--lvector_dir', help='Where to find images', type=str, default="out/seeds", required=False, metavar='DIR')
@click.option('--label_dir', help='Where to save labels', type=str, default="out/labels", required=False, metavar='DIR')
def train_svm(
   lvector_dir: str,
   label_dir: str
):
    """Train SVM on latent vectors"""
    start_time = perf_counter()
    os.makedirs(label_dir, exist_ok=True)

    # Data.
    latent_vectors = torch.load(lvector_dir+'/latent_vectors.pt').cpu().numpy()

    # Labels.
    labels = torch.load(label_dir+'/age_labels.pt').cpu().numpy()
    grouped_labels = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    labels = np.array(grouped_labels)[labels]
    count = np.bincount(labels)

    # # model = LinearSVC(verbose=2, max_iter=100000, tol=1e-5)
    # model = SVC(kernel='linear', decision_function_shape='ovr')

    # model.fit(latent_vectors, labels)
    # print(model.score(latent_vectors, labels))
    # print(model.coef_)
    # print(model.coef_.shape)

    n_classes = len(np.unique(labels))
    coef = np.zeros((n_classes, latent_vectors.shape[1]))
    intercept = np.zeros(n_classes)
    accuracy = 0

    for class_idx, class_ in enumerate(np.unique(labels)):
        y_binary = np.where(labels == class_, 1, -1)

        svm_binary = SVC(kernel='linear',decision_function_shape='ovr')
        svm_binary.fit(latent_vectors, y_binary)

        acc = svm_binary.score(latent_vectors, y_binary)
        accuracy += acc*count[class_idx]
        print("Accuracy: %f" % acc)

        coef[class_idx] = svm_binary.coef_.ravel()
        intercept[class_idx] = svm_binary.intercept_

        print("Time: %d" % (perf_counter() - start_time))

    print("Accuracy: %f" % (accuracy/len(labels)))
    print(coef)
    print(intercept)
    print("Time Elapsed: %d" % (perf_counter() - start_time))

    # save the coefficients as a tensor
    torch.save(coef, 'out/vectors/age_coef.pt')

    
   

#----------------------------------------------------------------------------

if __name__ == "__main__":
    train_svm() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
