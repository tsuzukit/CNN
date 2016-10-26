import svhn as svhn
import model as model
import numpy as np

if __name__ == '__main__':

    svhn = svhn.SVHN()
    images, labels = svhn.get_reformatted_dataset()

    # create model instance
    cnnModel = model.CNN()

    # train model
    cnnModel.fit(images, labels)

    # test model
    test_images, test_labels = svhn.get_reformatted_test_dataset()
    cnnModel.test(test_images, test_labels)

    # predict
    test_image, test_label = np.vsplit(test_images, [10])
    cnnModel.predict(test_image)

