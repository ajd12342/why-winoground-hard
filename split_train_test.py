train_idxs = [307, 222, 201, 327, 321, 73, 74, 344, 229, 267, 78, 70, 41, 50, 87, 93, 305, 376, 334, 333, 342, 86, 337, 224, 151, 95, 98, 325, 310, 180, 353, 263, 343, 76, 221, 394, 77, 75, 354, 340, 335, 23, 369, 367, 291, 358, 2, 385, 357, 356, 316, 27, 10, 336, 208, 209, 174, 245, 217, 62, 110, 203, 32, 107, 320, 314, 164, 83, 246, 54, 339, 61, 247, 31, 274, 386, 13, 326, 237, 90, 399, 315, 80, 301, 159, 17, 324, 84, 163, 264, 256, 363, 250, 173, 142, 19, 166, 251, 146, 235, 72, 236, 395, 131, 148, 309, 15, 177, 158, 170, 69, 313, 361, 14, 9, 154, 227, 272, 249, 210, 140, 213, 332, 233, 290, 6, 7, 155, 226, 68, 302, 132, 285, 231, 157, 306, 371, 362, 149, 234, 81, 212, 243, 16, 360, 49, 286, 161, 255, 352, 167, 143, 8, 11, 384, 3, 296, 175, 228, 5, 378, 160, 176, 289, 165, 121, 297, 59, 261, 24, 122, 239, 47, 185, 382, 269, 294, 317, 53, 191, 52, 390, 116, 311, 179, 29, 346, 129, 112, 268, 79, 240, 118, 108, 391, 197, 341, 374, 199, 120, 304, 105, 128, 60, 0, 30, 109, 279, 292, 288, 188, 282, 244, 39, 21, 204, 1, 308, 192, 106, 45, 300, 57, 276, 370, 248, 225, 281, 26, 258, 43, 127, 205, 200, 51, 253, 260, 186, 100, 266, 63, 114, 182, 48, 273, 65, 193, 33, 278, 330, 139, 277, 134, 329, 252, 28, 195, 312, 156, 25, 397, 58, 22, 85, 104, 280, 4, 242, 230, 126, 145, 20, 196, 117, 125, 44, 34, 46, 137, 184, 328, 298, 270, 64, 206, 183, 119, 331, 259, 366, 135, 82, 396, 207, 295, 35, 71, 130, 271, 138]
test_idxs = [262, 364, 350, 355, 275, 219, 372, 383, 303, 133, 150, 96, 351, 88, 323, 348, 218, 380, 345, 338, 67, 388, 393, 89, 136, 202, 254, 238, 55, 99, 318, 349, 319, 379, 91, 171, 144, 373, 18, 178, 216, 220, 147, 389, 168, 211, 141, 322, 293, 12, 368, 153, 365, 172, 257, 265, 162, 152, 169, 40, 187, 392, 214, 66, 113, 190, 56, 102, 299, 123, 223, 215, 94, 97, 241, 42, 124, 189, 92, 36, 198, 375, 387, 181, 381, 232, 284, 287, 101, 283, 347, 194, 37, 377, 115, 359, 111, 398, 38, 103]

import pickle as pkl

winoground_path = '/saltpool0/data/layneberry/WinoGround/'
file_to_split = 'clip_variants_feats.pkl'

train_set = []
test_set = []

all_data = pkl.load(open(winoground_path+file_to_split,'rb'))

for d in range(len(all_data)):
    if d in train_idxs:
        train_set.append(all_data[d])
    elif d in test_idxs:
        test_set.append(all_data[d])
    else:
        print('HEY! Index', d, 'isn\'t in any set!')

pkl.dump(train_set, open(winoground_path+file_to_split[:-4]+'.pkl', 'wb'))
pkl.dump(test_set, open(winoground_path+file_to_split[:-4]+'.pkl', 'wb'))
