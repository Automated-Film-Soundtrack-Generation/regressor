from essentia.standard import (
    MonoLoader,
    TensorflowPredictVGGish,
    TensorflowPredictMusiCNN,
    TensorflowPredict,
)
from essentia import Pool
import pandas as pd
import numpy as np

import os

MODELS_HOME = "./models"
FILES_DIR = "./wavs"

class Predictor():
    def __init__(self):
        """Load the model into memory and create the Essentia network for predictions"""

        self.musicnn_graph = str(MODELS_HOME + "/" + "msd-musicnn-1.pb")
        self.vggish_graph = str(MODELS_HOME + "/" + "audioset-vggish-3.pb")

        self.sample_rate = 16000

        self.pool = Pool()
        self.loader = MonoLoader()
        self.embeddings = {
            "msd-musicnn": TensorflowPredictMusiCNN(
                graphFilename=self.musicnn_graph, output="model/dense/BiasAdd"
            ),
            "audioset-vggish": TensorflowPredictVGGish(
                graphFilename=self.vggish_graph, output="model/vggish/embeddings"
            ),
        }

        self.input = "flatten_in_input"
        self.output = "dense_out"
        # Algorithms for specific models.
        self.classifiers = {}

        datasets = ("emomusic", "deam", "muse")
        # datasets = ("deam", "muse")
        for dataset in datasets:
            for embedding in self.embeddings.keys():
                classifier_name = f"{dataset}-{embedding}"
                graph_filename = str(MODELS_HOME + "/" + f"{classifier_name}-1.pb")

                print(classifier_name)

                self.classifiers[classifier_name] = TensorflowPredict(
                    graphFilename=graph_filename,
                    inputs=[self.input],
                    outputs=[self.output],
                )

    def predict(self, audio, embedding_type="msd-musicnn", dataset="muse"):
        """Run a single prediction on the model"""

        assert audio, "Specify either an audio filename or a YouTube url"

        # title = audio.name

        print("loading audio...")
        self.loader.configure(sampleRate=self.sample_rate, filename=str(audio))
        waveform = self.loader()

        embeddings = self.embeddings[embedding_type](waveform)

        # resize embedding in a tensor
        embeddings = np.expand_dims(embeddings, (1, 2))
        self.pool.set(self.input, embeddings)

        classifier_name = f"{dataset}-{embedding_type}"
        results = self.classifiers[classifier_name](self.pool)[self.output]
        results = np.mean(results.squeeze(), axis=0)

        # Manual normalization (1, 9) -> (-1, 1)
        results = (results - 5) / 4

        valence = results[0]
        arousal = results[1]

        return results


if __name__=='__main__':
    pred = Predictor() 

    print("File count: ", len(os.listdir(FILES_DIR)))

    inputs = []
    preds = []

    iteration = 1

    for file_name in os.listdir(FILES_DIR):
        if iteration > 100:
            break

        print(f"File {iteration}")
        inputs.append(int(file_name[1]))

        results = pred.predict(FILES_DIR + '/' + file_name)
        
        valence = results[0]
        arousal = results[1]

        if valence > 0.0 and arousal > 0.0:
            preds.append(1)
        elif valence < 0.0 and arousal > 0.0:
            preds.append(2)
        elif valence < 0.0 and arousal < 0.0:
            preds.append(3)
        else:
            preds.append(4)

        print(f"Quad: {inputs[-1]}, Predicted: {valence, arousal}")
        iteration += 1

    correctness = sum([1 if inputs[i] == preds[i] else 0 for i in range(len(inputs))])
    accuracy = correctness / len(inputs)

    print(f"Accuracy: {accuracy}%.")