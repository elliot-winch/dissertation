import cv2
import numpy as np

img = cv2.imread('../test_data/Test/Failure/2ac65e3c-c84f-7029-db2e-00000b0c33b8_Male_58_Phakic_Failure.png')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
