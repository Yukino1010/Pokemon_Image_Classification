# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:22:44 2021

@author: s1253
"""

import pandas as pd
import csv
import os
import cv2
from skimage import color

def load_pokemon_dict():

    pokemon_path = r"poke_data\poke_type.csv"
    pokemon_dict = {}
    with open(pokemon_path,"r") as f:
        reader = csv.reader(f)
        next(reader, None) 
        
        for row in reader:
            pkm_id = int(row[0])
            type_slot = int(row[2])
            pkm_type = int(row[1])
            
            if pkm_id not in pokemon_dict:
                pokemon_dict[pkm_id] = {"type1": None, "type2": None}
                
            if type_slot == 1:
                pokemon_dict[pkm_id]["type1"] = pkm_type
                
            elif type_slot == 2:
                pokemon_dict[pkm_id]["type2"] = pkm_type
                
            else:
                raise ValueError("Unexpected type slot value")
    return pokemon_dict


def load_dataframe(gen_folders):
    pkm_dict = load_pokemon_dict()
    df_dict = {"id" : [], "gen":[],"type1": [], "type2": [], "sprite" : []}
    sprites_folder = "centered-sprites"
    
    for gen, max_pkm in gen_folders.items():
        gen_folder = os.path.join(sprites_folder,gen)
        
        for pkm_id in range(1,max_pkm+1):
            image_file = "{id}.png".format(id=pkm_id)
            image_path = os.path.join(gen_folder,image_file)
            image = cv2.imread(image_path)
            
            image = color.rgb2hsv(image)
            df_dict["id"].append(pkm_id)
            df_dict["type1"].append(pkm_dict[pkm_id]["type1"])
            df_dict["type2"].append(pkm_dict[pkm_id]["type2"])
            df_dict["sprite"].append(image)
            df_dict["gen"].append(gen)
            
            
    return pd.DataFrame.from_dict(df_dict)











