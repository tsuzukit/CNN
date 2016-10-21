import mnist as mnist
import model as model

if __name__ == '__main__':

    print '# MNIST training data'
    mnist_training = mnist.MNIST(mnist.MNIST.TYPE_TRAINING)
    images, labels = mnist_training.get_reformatted_dataset()

    nnModel = model.Model()
    nnModel.fit(images, labels)

