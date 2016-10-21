import mnist as mnist
import model as model

if __name__ == '__main__':

    # MNIST prediction
    mnist_training = mnist.MNIST(mnist.MNIST.TYPE_TRAINING)
    images, labels = mnist_training.get_reformatted_dataset()
    cnnModel = model.CNN()
    cnnModel.fit(images, labels)

