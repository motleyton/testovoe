import logging
import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


class Pipeline:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.encoder = OrdinalEncoder()
        self.normalizer = StandardScaler()
        self.data = None
        self.__set_log_params()
        self.__load_config()

    def __drop_na(self):
        self.data.dropna(subset=self.config['drop_na_features'], inplace=True)

    def __fit_encoder(self):
        logging.info("fitting encoder...")
        self.encoder.fit(self.data[self.config['features_to_encode']])

    def __encode_data(self):
        logging.info("encoding data...")
        features_to_encode = self.config['features_to_encode']
        self.data[features_to_encode] = self.encoder.transform(self.data[features_to_encode])

    def __fill_with_median(self):
        for feature in self.config['fill_with_median']:
            median = self.data[feature].median()
            self.data[feature].fillna(median, inplace=True)

    def __fill_with_zero(self):
        for feature in self.config['fill_with_zero']:
            self.data[feature].fillna(0, inplace=True)

    def __load_config(self, path: str = 'config.yaml') -> None:
        try:
            with open(path) as f:
                self.config = yaml.safe_load(f)

        except FileNotFoundError:
            logging.error("Config file not found")
            exit()

    def __set_log_params(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s: %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def __fit_normalizer(self):
        logging.info("fitting normalizer")
        self.normalizer.fit(self.data[self.config['features_to_normalize']])

    def __normalize_data(self):
        features_to_normalize = self.config['features_to_normalize']
        self.data[features_to_normalize] = pd.DataFrame(self.normalizer.transform(self.data[features_to_normalize]))

    def __drop_garbage(self):
        self.data.drop(self.config['drop_features'], axis=1, inplace=True)
        logging.info("garbage dropped")

    def __preprocessing(self):
        logging.info("preprocessing data...")
        self.__drop_garbage()
        self.__drop_na()
        self.__fill_with_median()
        self.__fill_with_zero()

    def __x_y_define(self):
        self.y = self.data[self.config['y']]
        self.X = self.data.drop(self.config['y'], axis=1)

    def __train_model(self):
        logging.info("training model...")
        self.model.fit(self.X, self.y)

    def __save_models(self):
        logging.info("saving model...")
        with open(self.config['model_file_path'], 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.config['encoder_file_path'], 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.config['normalizer_file_path'], 'wb') as f:
            pickle.dump(self.model, f)

    def train(self, data: pd.DataFrame):
        self.data = data
        self.__preprocessing()
        self.__fit_encoder()
        self.__encode_data()
        self.__fit_normalizer()
        self.__normalize_data()
        self.__x_y_define()
        self.__train_model()
        self.__save_models()

    def predict(self, data: pd.DataFrame):
        self.data = data
        self.__preprocessing()
        self.__encode_data()
        self.__normalize_data()
        self.__x_y_define()
        preds = self.model.predict(self.X)
        return preds


if __name__ == '__main__':
    data = pd.read_csv("data/dataset.csv", sep=";")
    pipe = Pipeline()
    pipe.train(data)
    data = pd.read_csv("data/dataset.csv", sep=";")
    print(pipe.predict(data))
