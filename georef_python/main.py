import sys
import ImageManager

if __name__ == '__main__':
	ImageManager = ImageManager.ImageManager(sys.argv[1])
	ImageManager.detect_lines()
