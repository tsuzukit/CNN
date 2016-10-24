import mnist as mnist
import svhn as svhn
import model_mnist as model_mnist
import model as model

if __name__ == '__main__':

    svhn = svhn.SVHN()
    images, labels = svhn.get_reformatted_dataset()
    cnnModel = model.CNN()
    cnnModel.fit(images, labels)

    # mnist_training = mnist.MNIST(mnist.MNIST.TYPE_TRAINING)
    # images, labels = mnist_training.get_reformatted_dataset()
    # cnnModel = model_mnist.CNN()
    # cnnModel.fit(images, labels)

