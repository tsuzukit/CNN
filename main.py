import svhn as svhn
import model as model

if __name__ == '__main__':

    svhn = svhn.SVHN()
    images, labels = svhn.get_reformatted_dataset()
    cnnModel = model.CNN(dropout=0.8)

    # # train model
    cnnModel.fit(images, labels)

    # test model
    test_images, test_labels = svhn.get_reformatted_test_dataset()
    cnnModel.predict(test_images, test_labels)

