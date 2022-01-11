import cv2
import json
from jtop import jtop
import numpy as np
import onnx
import onnxruntime
import os
from PIL import Image
from shutil import copyfile
import sys
import time
from utils import format_results, heat_check, load_config, load_logger, tiles_location_gen

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
    if not os.path.exists(config["results"]["path"]):
        os.makedirs(config["results"]["path"])

    plots_path = os.path.join(config["results"]["path"], "./predictions")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)    

    jetson = jtop()
    jetson.start()

    print("----- Lux Ship Detection -----")

    # ----------------------------------------
    # Load model
    # ----------------------------------------
    onnx_model = onnx.load(config["model"]["onnx"]["path"])
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(config["model"]["onnx"]["path"])
    print("Model loaded on the " + onnxruntime.get_device())
    logger.info("Model loaded on the " + onnxruntime.get_device())

    # ----------------------------------------
    # Stats Keeping
    # ----------------------------------------
    times = []
    initial_time = time.time()
    end_time = time.time()
    running = True

    print("Start Analyzing")
    logger.info("Start Analyzing")

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
        files = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder)]
        files = sorted(files)
        file_path = files[-1]

        print(f"Analyzing {file_path}")
        logger.info(f"Analyzing {file_path}")

        # Track time
        iteration_start_time = time.time()

        # read image
        image = Image.open(file_path)
        img_size = image.size
        tile_size = (3, config["images"]["tile"]["width"], config["images"]["tile"]["height"])

        results = {}
        tile_locations = tiles_location_gen(img_size, tile_size[1:], 0)

        count = 0
        # Tile image
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
            '''if len(ort_outs[0]) > 0:
                results[count] = format_results(ort_outs)'''

            del ort_outs

            # Increment
            count += 1

        # Save predictions
        '''pred_path = os.path.join(
            os.path.join(config["results"]["path"], "predictions"),
            file_path.split("/")[-1].replace(config["images"]["type"], ".json")
        )
        print(f"Save results to {pred_path}")
        with open(pred_path, 'w') as fp:
            json.dump(results, fp)

        del pred_path
        del results'''

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

