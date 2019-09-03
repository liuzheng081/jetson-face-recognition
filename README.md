Simple face recognition app for Jetson Nano, inspired by [Adam Geitgey's tutorial](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd).

- Recognises faces using [face_recognition](https://github.com/ageitgey/face_recognition)
- Uses [hnswlib](https://github.com/nmslib/hnswlib) to quickly compare face embeddings
- Streams video output with [flask](https://github.com/pallets/flask)
- Saves new face embeddings and logs seen faces to MongoDB