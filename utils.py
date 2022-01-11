import logging
import time
import yaml


def heat_check(jetson, max_temp):
    """
    Checks if the Nano is getting too hot. If temperature reaches
    the maximum allowed, computation is stopped for a minute.
    ---
    max_temp: the maximum allowed temperature
    """
    too_hot = False
    temps = ["AO", "AUX", "CPU", "GPU", "thermal", "PLL"]
    for temp in temps:
        key = "Temp " + temp
        if key in jetson.stats and jetson.stats[key] >= max_temp:
            too_hot = True

    # If device is too hot
    if too_hot:
        # Take a 60 sec. pause
        print("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)


def load_config(config_path):
    config = yaml.safe_load(open(config_path))
    return config


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


def format_results(results):
    return {
        "bbox": [bbox.tolist() for bbox in results[0]],
        "label": [score.tolist() for score in results[1]],
        "score": [score.tolist() for score in results[2]]
    }
    return results


def tiles_location_gen(img_size, tile_size, overlap):
    """
    Generates location of tiles after splitting the given image according the tile_size and overlap.
    ---
    Args:
        img_size (int, int): size of original image as width x height.
        tile_size (int, int): size of the returned tiles as width x height.
        overlap (int): The number of pixels to overlap the tiles.
    Returns:
        A list of points representing the coordinates of the tile in xmin, ymin, xmax, ymax.
    """
    locations = []
    tile_width, tile_height = tile_size
    img_width, img_height = img_size
    h_stride = tile_height - overlap
    w_stride = tile_width - overlap
    for h in range(0, img_height, h_stride):
        for w in range(0, img_width, w_stride):
            xmin = w
            ymin = h
            xmax = min(img_width, w + tile_width)
            ymax = min(img_height, h + tile_height)
            locations.append([xmin, ymin, xmax, ymax])
    return locations
