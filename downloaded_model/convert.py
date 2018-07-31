from model import get_testing_model

def gen_trained_model():
    model = get_testing_model()
    model.compile
    model.load_weights('weights.0100.h5')
    model.save('keras_openpose_trained_model.hd5')

if __name__ == '__main__':
    gen_trained_model()