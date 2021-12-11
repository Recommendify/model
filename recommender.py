import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import requests
from requests.auth import HTTPBasicAuth


class Recommender(object):
    """ Spotify Recommender """

    def __init__(self, dataset_path):
        print("Recommender Class Instantiated")
        self.client_id = "Spotify Client ID"
        self.client_secret = "Spotify Client Secret"
        self.dataset = pd.read_csv(dataset_path)
        self.exist_in_dataset = False
        self._api_token = ""
        self._api_timestamp = 0
        self.cols = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'year', 'popularity'
        ]

    def _getApiToken(self):
        ctime = datetime.datetime.now()

        # if API token has been issued over half an hour, request a new one
        if ctime.timestamp() - self._api_timestamp >= 1800:
            res_api = requests.post(
                "https://accounts.spotify.com/api/token",
                data={"grant_type": "client_credentials"},
                auth=HTTPBasicAuth(self.client_id, self.client_secret)
            )

            if res_api.status_code == 200:
                token_object = res_api.json()
                token = token_object["access_token"]
                ctime = datetime.datetime.now()
                self._api_token = token
                self._api_timestamp = ctime.timestamp()
            else:
                raise Exception('API Token Request Failed')

        print('Current API Token: {}'.format(self._api_token))
        print('API Issue Timestamp: {}'.format(self._api_timestamp))

        return self._api_token

    def findSong(self, id):
        token = self._getApiToken()
        url_song_features = "https://api.spotify.com/v1/audio-features/"
        url_tracks = "https://api.spotify.com/v1/tracks/"
        metadata_cols = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms'
        ]

        res_song_features = requests.get(
            url_song_features + id,
            headers={"Authorization": "Bearer " + token}
        )

        res_tracks = requests.get(
            url_tracks + id,
            headers={"Authorization": "Bearer " + token}
        )

        json_song_features = res_song_features.json()
        json_tracks = res_tracks.json()

        features = {}
        features.update(dict(((key, json_song_features[key]) for key in metadata_cols)))
        features["year"] = int(json_tracks["album"]["release_date"][:4])
        features["popularity"] = json_tracks["popularity"]
        features["name_track"] = json_tracks["name"]
        features["name_album"] = json_tracks["album"]["name"]
        # Artist might be multiple
        features["name_artist"] = json_tracks["artists"][0]['name']

        return pd.DataFrame(features, index=[0])

    def collectSong(self, id, dataset):
        try:
            data = dataset[dataset['id'] == id].iloc[0]
            self.exist_in_dataset = True
            return data
        except IndexError:
            data = self.findSong(id)
            return data

    def getVector(self, song_lst, dataset):
        self.exist_in_dataset = False
        vectors = []

        for song in song_lst:
            data = self.collectSong(song, dataset=dataset)
            if data is None:
                print('{} is not found.'.format(data['name']))
                continue
            vectors.append(data[self.cols].values)

        tmp = np.mean(np.array(list(vectors)), axis=0)
        if tmp.shape != (1, 14):
            tmp = tmp.reshape(1, 14)

        return pd.DataFrame(tmp, columns=self.cols)

    def changeDictList(self, dict_list):
        new = {}
        for key in dict_list[0].keys():
            new[key] = []

        for dictionary in dict_list:
            for key, value in dictionary.items():
                new[key].append(value)

        return new

    def minmaxTransform(self, data, dataset):
        for column in data[self.cols].columns:
            data[column] = (data[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())
        return data[self.cols]

    def recommend(self, song_lst, specific_year=None, number=10):
        require_data = ['name', 'year', 'artists', 'id']

        song_lst_mean = self.getVector(song_lst, dataset=self.dataset)

        if specific_year is None:
            data_transform = self.minmaxTransform(self.dataset[self.cols], self.dataset[self.cols])
        else:
            dataset = self.dataset[self.dataset['year'].isin(specific_year)]
            df = dataset[self.cols]
            data_transform = self.minmax_transform(df, df)

        song_lst_mean = self.minmaxTransform(song_lst_mean, self.dataset[self.cols])
        d = cdist(song_lst_mean, data_transform, 'cosine')
        i = list(np.argsort(d)[
                 :, 1:number + 1][0]) if self.exist_in_dataset else list(np.argsort(d)[:, :number][0])

        results = self.dataset.iloc[i]

        return results[require_data].id.to_list()


if __name__ == "__main__":
    rc = Recommender('data.csv')
    print(rc.recommend(["5HCyWlXZPP0y6Gqq8TgA20"]))
