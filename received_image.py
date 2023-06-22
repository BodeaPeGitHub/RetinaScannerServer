class ReceivedImage:

    def __init__(self, image, position, client):
        self.__image = image
        self.__position = position
        self.__client = client
        self.__prediciton = None
        self.__image_with_explainale = None

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        self.__image = image

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        self.__position = position

    @property
    def client(self):
        return self.__client

    @client.setter
    def client(self, client):
        self.__client = client

    @property
    def prediction(self):
        return self.__prediction

    @prediction.setter
    def prediction(self, prediction):
        self.__prediction = prediction

    @property
    def image_with_explainable(self):
        return self.__image_with_explainable

    @image_with_explainable.setter
    def image_with_explainable(self, image_with_explainable):
        self.__image_with_explainable = image_with_explainable