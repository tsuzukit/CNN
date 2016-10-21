import mnist as mnist
import svhn as svhn
import model as model

if __name__ == '__main__':

    svhn = svhn.SVHN()
    svhn.show_as_image()

    # mnist_training = mnist.MNIST(mnist.MNIST.TYPE_TRAINING)
    # mnist_training.show_as_image()

    # images, labels = mnist_training.get_reformatted_dataset()
    # cnnModel = model.CNN()
    # cnnModel.fit(images, labels)

