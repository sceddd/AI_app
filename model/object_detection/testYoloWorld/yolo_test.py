import os
from pathlib import Path
import argparse
import ultralytics
from humanfriendly.terminal import output

novel_classes = ['person', 'car', 'motorbike', 'snail']
rare_classes = ['snail', 'chicken', 'glasses']


def test_yolo(args):
    HOME = Path(os.getcwd())
    list_images = os.listdir(str(Path("photos")/ args.type))
    dir = HOME / "photos" / args.type
    YOLO_WORLD = ultralytics.YOLO(args.yolo_cp)
    classes = novel_classes if str(args.type).startswith('novel') else rare_classes
    YOLO_WORLD.set_classes(classes)

    results = [YOLO_WORLD(str(dir / image), conf=args.conf) for image in list_images]
    for r in results:
        r[0].save(str(HOME / 'output' / os.path.basename(r[0].path)))


def get_parse():
    arg = argparse.ArgumentParser()
    arg.add_argument("--yolo_cp", type=str, help="yolo model")
    arg.add_argument("--type", type=str, help="type of classes")
    arg.add_argument("--conf", type=float, help="confidence", default=0.2)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_parse()
    test_yolo(args)
    print("Done")
