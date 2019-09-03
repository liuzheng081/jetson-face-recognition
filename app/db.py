import pickle

import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
import hnswlib


class DB:
    def __init__(self):
        self.mongo_client = MongoClient(
            host='localhost', port=27017,
            readPreference='secondaryPreferred'
        )
        self.collection = self.mongo_client.fifa_players
        self.logs = self.mongo_client.fifa_logs

        self.dim = 128
        self.max_elements = 1000
        self.index, self.known_face_metadata = self.load_data()

    def load_data(self):
        index = hnswlib.Index(space='cosine', dim=self.dim)
        metadata = list(self.collection.find({}, {'_id': 0}))

        index.init_index(max_elements=self.max_elements)
        index.set_num_threads(4)
        index.set_ef(10)
        index.add_items(np.array([i['embedding'] for i in metadata]))

        return index, metadata

    def register_new_face(self, face_encoding, photo):
        id = len(self.index.get_ids_list())
        photo = Binary(pickle.dumps(photo, protocol=2))
        new_user = dict(
            id=id,
            name=id,
            embedding=list(face_encoding),
            photo=photo
        )
        self.collection.insert(new_user)
        self.known_face_metadata.append(dict(id=id, name=id, embedding=list(face_encoding)))

        self.index.add_items([face_encoding])
        return id

    def lookup_known_face(self, face_encoding):
        user_data = None

        if len(self.known_face_metadata) == 0:
            return None, None

        label, distance = self.index.knn_query(face_encoding, k=1)

        if distance[0][0] < 0.08:
            user_data = self.known_face_metadata[label[0][0]]
            user_data['distance'] = round(float(distance[0][0]), 2)

        return user_data, distance

    def update_photo(self, id, photo):
        photo = Binary(pickle.dumps(photo, protocol=2))
        self.collection.update_one(
            {'id': id, 'updated': {'$exists': False}},
            {'$set': {'photo': photo, 'updated': 1}},
            upsert=False
        )

    def save_logs(self, data):
        self.logs.insert(data)
