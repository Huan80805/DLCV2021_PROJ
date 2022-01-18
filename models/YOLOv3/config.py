class TrainConfig():
    def __init__(self):
        self.seed = 7
        self.img_size = 512
        self.lr = 1e-5
        self.weight_decay = 5e-7
        self.window_size = 20
        self.lr_decay = 0.9