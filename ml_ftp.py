import ftplib
from io import BytesIO
import jetson.inference
import jetson.utils
import json
from jtop import jtop
import logging
import numpy as np
import os
from PIL import Image
import time
import yaml


def get_image(image, width, height):
    """
    Converts an image into a tensor and tiles it to smaller patches.
    ---
    image (PIL image): The image in PIL format
    width (int): width of each tile 
    height (int): height of each tile 
    """
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
    img (tensor): the image as a tensor of shape (channels, width, height)
    width (int): the width of every patch
    height (int): the height of every patch
    """
    return img.data.unfold(0, 3, 3).unfold(1, width, height).unfold(2, width, height).squeeze()


def reconstruct(img, tiles):
    """
    Reconstruct an image based on tiles
    ---
    img (tensor): the original image
    tiles (tensor): tiles in the shape (1, rows, cols, channels, width, height)
    """
    return tiles.permute(2, 0, 3, 1, 4).contiguous().view_as(img)


def load_model(model_path):
    """
    Loads the PyTorch model.
    ---
    model_path (string): path to the .pth file
    """
    # Get model from PyTorch
    model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)

    # Load from saved weights
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    # Return model
    return model


def load_logger(logger_path):
    """
    Loads the logger
    ---
    config_path (string): path to the logging file
    """
    # create logger with 'spam_application'
    logger = logging.getLogger('Lux')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger


def load_config(config_path):
    """
    Loads the configuration file
    ---
    config_path (string): path to the configuration file
    """
    config = yaml.safe_load(open(config_path))
    return config


def heat_check(jetson, max_temp):
    """
    Checks if the Nano is getting too hot. If temperature reaches
    the maximum allowed, computation is stopped for a minute.
    ---
    jetson (JTOP): jtop Object
    max_temp (int): the maximum allowed temperature
    """
    if jetson.stats["Temp AO"] >= max_temp or \
            jetson.stats["Temp CPU"] >= max_temp or \
            jetson.stats["Temp GPU"] >= max_temp or \
            jetson.stats["Temp PLL"] >= max_temp or \
            jetson.stats["Temp thermal"] >= max_temp:
        # Take a 60 sec. pause
        logger.info("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)


def connect_ftp(FTP_HOST, FTP_USER, FTP_PASS):
    """
    Connect to the FTP server
    ---
    FTP_HOST (string): host adresse of the server
    FTP_USER (string): username
    FTP_PASS (string): password
    """
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    ftp.cwd('files')
    # force UTF-8 encoding
    ftp.encoding = "utf-8"
    return ftp


def log(logger, message, console=True):
    logger.info(message)
    if console:
        print(message)


logger = load_logger("logs.log")

if __name__ == "__main__":
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

    # Create predictions folder if it doesn't exist
    predictions_path = os.path.join(config["results"]["path"], "./predictions")
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    # ----------------------------------------
    # Load model
    # ----------------------------------------
    onnx_model = onnx.load(config["model"]["onnx"]["path"])
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(config["model"]["onnx"]["path"])
    print("Model loaded on the " + onnxruntime.get_device())
    logger.info("Model loaded on the " + onnxruntime.get_device())

    # ----------------------------------------
    # Jetson Nano
    # ----------------------------------------
    jetson = jtop()
    jetson.start()

    print("----- Lux Ship Detection -----")

    # Keep track of time
    initial_time = time.time()
    start_time = time.time()
    end_time = time.time()
    running = True

    log(logger, "Start Analyzing")
    while running:
        # If Nano gets too hot
        heat_check(jetson, config["inference"]["max_temperature"])

        # Limits execution time
        # For analysis purpose, remove in production
        if (time.time() - initial_time) > config["inference"]["execution_time"]:
            running = False
            break

        # Respect interval limit
        print(f"Waiting {config['inference']['intervals']} seconds")
        time.sleep(config["inference"]["intervals"])

        # Retrieve files
        ftp = connect_ftp(
            config["ftp"]["host"], config["ftp"]["username"], config["ftp"]["password"])
        log(logger, "Connected to FTP")
        files = ftp.nlst()
        files.sort()
        file_path = files[-1]
        del files
        log(logger, f"Retrieving {file_path}")
        # Read image
        with open(file_path, "wb") as file:
            flo = BytesIO()
            ftp.retrbinary(f"RETR {file_path}", flo.write)
            flo.seek(0)
            image = Image.open(flo).convert("RGB")
            ftp.quit()
            del ftp
            del flo

            # Tile image
            img_size = image.size
            tile_size = (3, config["images"]["tile"]["width"], config["images"]["tile"]["height"])
            tile_locations = tiles_location_gen(img_size, tile_size[1:], 0)

            # Loop tiles
            results = {}
            count = 0
            for tile_location in tile_locations:
                # To numpy float array
                tile = np.asarray(image.crop(tile_location), dtype=np.float32)
                # Reshape to put num. channels first
                tile = np.transpose(tile, (2, 0, 1))
                
                # Pad image if too small
                if not tile.shape == tile_size:
                    zeros = np.zeros(tile_size, dtype=np.float32)
                    zeros[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = zeros
                    
                # Add dimension for 'batch size' dimension
                tile = np.expand_dims(tile, axis=0)

                # Compute
                ort_inputs = {ort_session.get_inputs()[0].name: tile}
                ort_outs = ort_session.run(None, ort_inputs)

                del ort_inputs

                # Save result
                if len(ort_outs[0]) > 0:
                    results[count] = format_results(ort_outs)

                del ort_outs

                # Increment
                count += 1

            # Save predictions
            pred_path = os.path.join(
                os.path.join(config["results"]["path"], "predictions"),
                file_path.split("/")[-1].replace(config["images"]["type"], ".json")
            )
            print(f"Save results to {pred_path}")
            with open(pred_path, 'w') as fp:
                json.dump(results, fp)

            del pred_path
            del results

            times.append(time.time() - iteration_start_time)
            print("%s seconds" % (time.time() - iteration_start_time))

    # Set current time
    end_time = time.time()

    # Save stats
    np.savetxt(os.path.join(config["results"]["path"], "times.txt"), times, delimiter=',')
    if os.path.isfile("logs.log"):
        copyfile("logs.log", os.path.join(config["results"]["path"], "logs.log"))
    if os.path.isfile("stats.log"):
        copyfile("stats.log", os.path.join(config["results"]["path"], "stats.log"))
    if os.path.isfile("config.yml"):
        copyfile("config.yml", os.path.join(config["results"]["path"], "config.yml"))    

