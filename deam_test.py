from predict import Predictor

import pandas as pd
import numpy as np
import os

pred = Predictor()

ANNOTATION_FILE="deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
MUSIC_DIR="deam/DEAM_audio/MEMD_audio"

annotations = pd.read_csv(ANNOTATION_FILE, dtype=np.float32)
print(annotations)

running_total_val = 0
running_total_asl = 0

iterations = 0

annotations_normalized = (annotations - annotations.min()) / (annotations.max() - annotations.min())
annotations_normalized["song_id"] = annotations["song_id"]

print(annotations_normalized)

for file_name in os.listdir(MUSIC_DIR):
    song_id = int(file_name.split(".")[0])

    if song_id > 2000:
        print("will add support later im lazy")

    else:
        file_name = MUSIC_DIR + '/' + file_name

        results = pred.predict(file_name, embedding_type="audioset-vggish", dataset="deam")

        actual_val = annotations_normalized.loc[annotations_normalized["song_id"]==song_id][" valence_mean"]
        actual_asl = annotations_normalized.loc[annotations_normalized["song_id"]==song_id][" arousal_mean"]

        actual_vstd = annotations_normalized.loc[annotations_normalized["song_id"]==song_id][" valence_std"]
        actual_astd = annotations_normalized.loc[annotations_normalized["song_id"]==song_id][" arousal_std"]

        print(f"{file_name} : ({results[0]}, {results[1]})\n    Actual : ({float(actual_val)}, {float(actual_asl)})")

        running_total_val += int(abs(results[0] - float(actual_val)) < float(actual_vstd))
        running_total_asl += int(abs(results[1] - float(actual_asl)) < float(actual_astd))

        iterations += 1


print(f"Accuracy:\n\t\tValence : {running_total_val / iterations}\n\t\tArousal : {running_total_asl / iterations}")


