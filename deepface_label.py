import os
import numpy as np
from deepface import DeepFace
import click

@click.command()
@click.option("--path", default="images", help="Path to the directory containing image files")
@click.option("--output-path", default="output.npy", help="Path to save the output file")
def extract_labels(path, output_path):
    data = []
    for file in os.listdir(path):
        # analyze image using DeepFace
        objs = DeepFace.analyze(img_path=os.path.join(path, file), actions=['age', 'gender', 'race', 'emotion'], silent=True)

        # extract relevant information
        age = objs[0]['age']
        gender = objs[0]['dominant_gender']
        race = objs[0]['dominant_race']
        emotion = objs[0]['dominant_emotion']

        print(f"NAME: {file},\t Age: {age},\t Gender: {gender},\t Race: {race},\t Emotion: {emotion}")
        # append to data list
        data.append([file, age, gender, race, emotion])

    # convert data to NumPy array
    data_arr = np.array(data)

    # check if output directory exists, create if it doesn't
    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save to .npy file
    np.save(output_path, data_arr)

if __name__ == '__main__':
    extract_labels()
