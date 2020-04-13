import sys
import argparse
from yolo import YOLO, detect_image, detect_video, detect_test_set

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    parser.add_argument(
        "--video", nargs='?', type=str, required=False, default='./video.mp4',
        help = "Video input path"
    )
    parser.add_argument(
        "--testset", nargs='?', type=str, required=False, default='./test.txt',
        help = "Text set input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        detect_image(YOLO())
    elif FLAGS.testset:
        detect_test_set(YOLO(), FLAGS.testset, FLAGS.output)
    elif FLAGS.video:
        detect_video(YOLO(), FLAGS.video, FLAGS.output)
    else:
        print("Must specify at least input_path. See usage with --help.")
