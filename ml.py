import jetson.inference
import jetson.utils
import json
from jtop import jtop
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys
import time
import torch
from torchvision import transforms, models
import yaml


def load_config(config_path):
    config = yaml.safe_load(open(config_path))
    return config


def get_image(image, width, height):
    # Image transformations
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    # Transform image
    image = trans(image)
    #image = torch.rand(3, 4304, 4304)
    # Split image into tiles
    tiles = tile(image, width, height)
    del image
    # Return
    return tiles


def tile(img, width, height):
    """
    Slices an image into multiple patches
    ---
    img: the image as a tensor of shape (channels, width, height)
    width: the width of every patch
    height: the height of every patch
    """
    return img.data.unfold(0, 3, 3).unfold(1, width, height).unfold(2, width, height).squeeze()


def reconstruct(img, tiles):
    """
    Reconstruct an image based on tiles
    ---
    img: the original image
    tiles: tiles in the shape (1, rows, cols, channels, width, height)
    """
    return tiles.permute(2, 0, 3, 1, 4).contiguous().view_as(img)


def load_model(model_path):
    # Get model from PyTorch
    model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)

    # Load from saved weights
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    # Return model
    return model


def load_logger(logger_path):
    # create logger with 'spam_application'
    logger = logging.getLogger('Lux')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger



def heat_check(jetson, max_temp):
    """
    Checks if the Nano is getting too hot. If temperature reaches
    the maximum allowed, computation is stopped for a minute.
    ---
    max_temp: the maximum allowed temperature
    """       
    if  jetson.stats["Temp AO"] >= max_temp or \
        jetson.stats["Temp CPU"] >= max_temp or \
        jetson.stats["Temp GPU"] >= max_temp or \
        jetson.stats["Temp PLL"] >= max_temp or \
        jetson.stats["Temp thermal"] >= max_temp:
        # Take a 60 sec. pause
        print("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)

logger = load_logger("logs.log")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        images_folder = args[0]
        if not os.path.isdir(images_folder):
            sys.exit("Invalid image folder path")
    else:
        sys.exit("Invalid image folder path")
        
    # ----------------------------------------
    # Configuration
    # ----------------------------------------
    config = load_config("./config.yml")    

    # ----------------------------------------
    # Results
    # ----------------------------------------
    # Create results folder if it doesn't exist

    if not os.path.exists(config["results"]["path"]):
        os.makedirs(config["results"]["path"])

    plots_path = os.path.join(config["results"]["path"], "./predictions")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)    

    jetson = jtop()
    jetson.start()

    print("----- Lux Ship Detection -----")

    # Tiling
    tile_width = config["images"]["tile"]["width"]
    tile_height = config["images"]["tile"]["height"]

    # Load model
    print("Loading model...")
    logger.info("Loading model...")
    model = load_model(config["model"]["torch"]["path"])
    # Send model to device
    model.eval().cuda()
    print("Model Loaded")
    logger.info("Model Loaded")
    
    # Clear Memory
    torch.cuda.empty_cache()

    # Keep track of time
    initial_time = time.time()
    start_time = time.time()
    end_time = time.time()
    running = True

    print("Start Analyzing")
    logger.info("Start Analyzing")
    while running:
        # If Nano gets too hot
        heat_check(jetson, config["inference"]["max_temperature"])
        # For analysis purpose, remove in production
        # Limits execution time
        if (time.time() - initial_time) > config["inference"]["execution_time"]:
            running = False
        if (end_time - start_time) >= config["inference"]["intervals"]:
            # Clear Memory
            torch.cuda.empty_cache()
            # Retrieve files
            files = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder)]
            files = sorted(files)
            file_path = files[-1]
            del files
            print("Analyzing " + file_path)
            logger.info("Analyzing " + file_path)
            # Read image
            image = Image.open(file_path).convert('RGB')
            # Get image
            batch = get_image(image, tile_width, tile_height)
            del image
            # Flatten patches
            batch = batch.contiguous().view((batch.size(0) * batch.size(1)), batch.size(2), batch.size(3), batch.size(4))
            # Send to GPU
            batch = batch.cuda()
            # Loop all tiles
            count = 1
            for image in batch:
                print(count)
                # Prediction
                #prediction = model(image.unsqueeze(0).cuda())
                prediction = model(image.unsqueeze(0))
                file_name = file_path.split('/')[-1].replace(config["images"]["type"], f"_{count}.pth")
                torch.save(prediction[0], config["results"]["path"] + "/predictions/" + file_name)
                # Clear Memory
                del prediction
                del file_name
                torch.cuda.empty_cache()
                # Increment count
                count += 1
            # Clear Memory
            del batch
            del count
            del file_path
            # Save Prediction
            print("Predictions saved")
            logger.info("Predictions saved")
            # Reset timer
            start_time = time.time()

        # Set current time
        end_time = time.time()

    print("Done !")
