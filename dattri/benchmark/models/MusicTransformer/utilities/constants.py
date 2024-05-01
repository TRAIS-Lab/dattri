# reference: https://github.com/gwinndr/MusicTransformer-Pytorch
# author: Damon Gwinn
import torch
import numpy as np

from dattri.benchmark.models.MusicTransformer.midi_processor.processor import (
    RANGE_NOTE_ON,
    RANGE_NOTE_OFF,
    RANGE_VEL,
    RANGE_TIME_SHIFT,
)

SEPERATOR               = "========================="

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

VOCAB_SIZE              = TOKEN_PAD + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4

SANITY_CHECK_LENGTH_256_TOTAL_5000 =\
    np.array([76740, 83580, 84355,    34, 24821, 58276,  4121, 81335, 29882,
       11972, 60929,  3750, 64208, 52782, 22161, 52758, 61376, 89006,
       51890, 40757, 19976, 74827,  4744, 28073, 74776, 25642, 81139,
       22622, 85722, 76016, 44965, 66916, 43709, 13890, 64119, 35452,
       56685, 87870, 71942,  2709, 37889, 34155, 61495, 64309, 79492,
       88786, 43763, 66400, 34571, 64067, 46310, 84033, 20547, 61757,
       42017, 14766, 87243, 87022, 48850, 40186, 31921, 67607, 13760,
       74611, 50220, 81469, 75927, 81745, 13824, 14337, 58922, 31975,
        3001, 43283, 41937,  1679, 10774, 74875, 62148, 23243, 89346,
       68886, 30941, 73423, 11687, 79956, 29789, 53060, 76074, 45237,
       59900, 45330, 35863, 60651, 75434, 47572, 71223, 28627, 27145,
       20105, 63622, 10170, 78078, 11522, 33356, 14781, 67309, 54015,
       22228, 55522,  7001, 85659, 31721, 39119,  2390, 56664, 20787,
       74385, 17721, 27179, 11135, 67398, 32966, 46470, 52644, 33647,
       68443, 80130, 34643,  6477, 56532, 10227, 59928, 16362, 27242,
       71663,  1499, 29270, 86076,  5666, 44956, 28560, 62512, 73114,
       22349, 82645, 32354, 66067, 28312, 63617, 78510, 24994, 13661,
       20965, 74729, 18516,  1466, 86861,  9120, 52387, 37956, 77482,
       21146, 82217, 62303, 55484, 37738,  8832,  2376, 51143, 63833,
       75031, 50199,  1107, 46554, 56739, 64081, 36037, 88823, 14197,
       21943, 46521, 42992, 55871, 86689, 13482, 88067, 35131, 54629,
       78057, 66926, 42769, 86839, 60369, 57748, 59423, 70178, 15374,
       48455, 45068,  3114, 79575, 41421, 60682,  3661, 60083, 15366,
       58804, 26486, 24714, 46274, 26281, 63176, 75462, 89628, 67879,
       65757, 18765, 27396, 32195, 29981, 35924, 71568, 39229, 28629,
       79305,  4250, 15670, 24147,  9002, 47709, 55798, 81587, 23669,
       64397, 67584,  1605, 26867, 26043, 18592, 27783, 43078, 27703,
       87482, 45289, 65394,  8466, 65854, 74035, 89675,  8514,  2814,
       77971,  1490, 58532, 68937, 18252, 64666, 29508, 35900, 14298,
       33000, 56737, 55383, 39314, 58411, 61045,    69, 27090, 55654,
       42190, 42276, 75097, 51574, 87602, 59032,  2630,  5583, 36353,
       30416, 12071, 53756, 57557, 15028, 32889, 39058,  9051, 39866,
       24616, 77350, 36187, 43820, 25287, 15572, 88236, 81475, 60657,
       14057, 39580, 73408,  3477, 86325, 60448, 76312, 69381, 12162,
       80234, 60148, 21921, 75834,  8178, 47611, 81185, 55190, 68770,
       45737, 69974, 62274, 68286, 86965, 14772, 39992, 23914, 15619,
       34262, 19950, 13286, 15793, 80809, 10160, 32648, 87864, 30015,
       39564, 68353, 88862,   409, 47069, 27712, 57289, 15998, 81411,
       69396,  6085, 87380,  4665, 24828, 87120, 81175, 24310, 59703,
       73678, 28420, 89094, 46035, 88254, 30146, 77769, 39467, 28768,
       65814, 77609, 31410, 19078, 35889, 63336, 45872, 21254, 11950,
       48262, 86175, 75590, 12654, 78884, 23549, 75992, 59669, 78783,
       24408, 12936, 20917, 55380, 34566, 39479, 34564, 12711, 21732,
       23422, 71661, 16641, 63754, 60934, 86515, 87312, 83462, 22327,
       49757, 83777, 65831, 70818, 72073, 80433, 37981, 14761, 67042,
       80878, 61616, 61552, 57621, 17422, 12499, 41022, 62073, 37814,
       72451, 66972, 49436, 30868, 48578,  1833, 73640, 39436,  3843,
       61640, 80398, 56915,  8681, 77537, 75285,  6971, 86362, 22487,
       49299, 26425, 36553, 43327, 65382, 38375, 14077, 44464, 34463,
       14459, 26479, 84054, 88032, 35263, 74792, 72863, 10175, 75498,
       73002, 10981, 25841, 88700, 15363, 46217, 19642, 71229, 38895,
       29621,  6995, 47775, 36375, 22022,  8793, 62977, 73764, 29596,
       45324,   579, 24606, 23015, 44795, 12973, 36304,  8306, 16870,
       20334, 68597, 45814, 38729, 64878, 42500, 29345,  7548, 81645,
       37665,  3216, 84604, 26735, 16311, 44351, 54399, 31765, 29952,
        8698, 23302, 53208, 45931, 11102,  8849, 46209,   127, 21598,
       65865,  3014,  8581, 48724, 52826, 29085, 41880, 70478,  5456,
       76788, 46611, 29818, 36421, 49434, 89031, 10433, 49267, 47379,
        9480, 12650, 59282, 59116, 36478, 40016, 34393, 15031, 67149,
       36727, 17757, 68179, 63824, 29961, 51032, 62526,  1940, 88212,
       36137, 27707, 68506, 39659, 33804, 13375, 50529, 69374, 23698,
       13859, 83069, 58028, 40387, 53467, 79487, 56672, 52160,  3582,
       11598, 13236, 87798, 82760,  6783, 87817, 52408, 54792, 46406,
       61604, 84674, 18945, 37897, 25616, 56125, 34650, 59003, 69662,
       25389, 87174, 78014, 54375, 44758, 39230, 66598, 47872, 73343,
       27858, 80192, 73215, 65341, 14044, 27597,  2364, 75727, 38976,
       27795, 25597, 23601, 63356, 43350,  7575, 47365, 78547, 17294,
       61287, 30771, 33035, 78647, 53099, 58419, 53315, 73825, 51424,
       80078, 81066,  1031, 82011, 64285, 57374, 14901, 81965, 34614,
        4884, 23487, 53347, 36271, 59708, 15041, 33901, 47875,  1236,
       66670, 71304,  4237, 89596, 82313, 58579, 58131, 28746,  8661,
       33868, 18608, 58628,  8247,   633, 23589,  6726, 46489, 75628,
       23584, 64707, 73974, 24877, 61932, 55498, 84636, 14278, 80452,
       78082, 58629, 55554, 35307, 83796, 42871,  7331, 74068, 18952,
       60314, 53112, 45889, 85745,  7350, 77142, 31804,  8313, 39723,
       45661, 41568, 71239, 47017, 20351, 77952, 66486, 25044, 75871,
       77538, 21175, 32036, 45426, 27616, 36872, 23501,  1088, 12715,
       88895, 84204, 15169, 80356, 84019, 55273, 78156, 29096, 10847,
       18457, 34506, 52895, 16702, 47200, 58588, 84481, 20953, 42836,
       11727, 18780, 69207, 17981, 29344, 22927,   722, 33677, 72927,
       21385, 31065, 75668, 23502, 80710, 25942, 35991, 42923, 82114,
        7271,  1475, 34211, 14917, 71679, 62841, 86008, 67552, 38730,
       12385, 13893, 26411, 69799, 67197,  4160,  9247, 88085, 38165,
       89055, 40077, 21283, 60617, 88984, 50223, 86332, 77200, 82808,
       11008, 68869, 35893,  5490, 72603, 65028, 86099, 52506, 69980,
       15983, 78567, 43009, 80161, 30278, 82613, 34652, 84026, 54698,
       69118, 29139, 23933, 87417, 64983, 82432, 81961, 46151, 10412,
       12859, 86246, 53322, 67973, 60974, 24903, 58403,  8775, 20230,
       13200, 73394, 78867, 89948, 57560, 51608, 79250, 33334, 80283,
       36739, 39018, 64791,  1614, 51409, 41548,  6837, 22567, 43773,
       86653, 47752, 72043, 78236,  1852, 87954, 80960, 38566,  2038,
       29047, 30433, 78730, 66230, 37871, 55073, 88575, 56002, 66273,
       24915,  2510, 79973, 33675, 84078, 41823, 86592, 14201, 27340,
       79611,  4030, 54782, 73477, 77103, 14017, 55413, 49542, 76795,
       86951, 77274,  9728,  5686,  3154, 75832, 65556, 62133, 59164,
       65990, 80352, 79461, 49471, 21853, 71624,  1462, 49552, 61020,
       43045, 46936, 15227, 20940, 60457, 64698, 72817,  2228, 28758,
       64924, 57092, 83931, 58206, 11168, 85377, 40998, 87235, 73123,
       44964, 41611, 41221,   410, 82467, 36385, 89931,  1020, 14537,
       15535, 69506, 43321,  8029, 52872, 12280,  3522, 23963, 64420,
        4350, 66823, 30088, 76006, 84668, 43791, 42254, 86125, 58264,
       86672, 64207, 77792, 77004, 33455, 15096,  7316, 78313, 77739,
       22682, 19179,  4818, 27230, 42816, 12878, 15785, 81153, 34832,
       50419, 88779, 76043, 27543, 42085, 51337, 39880, 10756, 70749,
       10579, 44112, 27833, 19684, 11855, 56534, 48281, 21040, 53065,
       45559, 69031,  3480, 84351, 38980, 61745,  3242, 56783, 53670,
       50808, 89731, 80619, 42805, 23716, 63626, 55279, 66427, 21776,
       30448, 37342, 53937, 39370, 87725, 33310, 66079,  8377, 13512,
       28687, 82802, 42434, 47787, 30719, 17330, 55171, 81924, 54210,
       23175, 73503, 41272, 76638, 69881, 85964, 79014, 53711, 10129,
        6181, 29053, 12117, 23472, 40330, 32096, 14109, 86427, 84029,
       44335, 44629, 75314, 79585, 27150, 10817,  8176, 81978, 85632,
        7522, 10853,  6376,  9162, 46343, 81712, 15259, 83340, 65227,
       26328, 17414, 29402, 78130, 32820, 67392, 54213, 28294, 34791,
       29691,  9842, 72503, 52340, 41834, 28741, 56090, 81287, 64787,
       45657, 23883, 74552, 76169, 29489, 74577, 32859, 32149, 33219,
       29446, 72957, 69908,  5193, 63918, 56693, 61290, 46272, 13352,
       13577, 62513, 66744,  4880, 49365, 56972, 68001,  6694,  9071,
       65949, 49384, 53428, 57802,  3198, 89199, 21908,  4274, 57601,
       61468, 19123, 14779, 11033, 11738, 30499, 78256, 23065, 12437,
       50707, 45893, 88764, 10490, 58896, 68650, 60958, 80945, 69913,
       38977, 14540, 20884, 65418, 66039, 75086, 87129, 19789, 57236,
       54349, 67121, 64770, 58423,  3370, 46837, 40427, 88464, 81734,
       31972, 60943, 19396, 17271, 81441, 42799, 74686, 87489, 74366,
       83622, 50166, 17223, 13269, 63207, 68775, 13860, 32587, 55642,
       85927, 33916, 36750, 30950, 64018,  1320,  1054,   305, 60340,
        9652,  1766, 26733, 10888, 37414, 10959, 70012, 16545, 23541,
       29308, 63241, 46172, 39841, 26172, 73420, 37010, 46600, 41490,
       36509, 33154, 39327, 66381, 22401, 60447, 64463, 24516,  4200,
       43951, 58912, 36734, 52994, 39075, 73226, 53050, 86627, 45989,
       87185, 67772, 67166,  7378, 60308, 19062, 72652, 40049, 88245,
       66421, 67070, 73562,  2575, 34960, 62549, 34807,  6659, 77009,
       66781, 51281, 75415, 47338, 64199, 35882, 39280, 35613, 53964,
       20594, 27043, 11132, 81214, 29629, 35550,  1716, 36867, 74707,
       75400, 10437, 61339, 41283, 30505, 40399, 73730, 89051, 66260,
       77082, 29809, 58015, 28334, 28395, 31027, 40142, 77691,  4311,
       29636,  4041, 68981, 58982, 69228, 42403, 84749, 41338, 76542,
       25888, 20795,  8255, 49230, 27015, 67159, 37201, 73656, 12114,
       53167, 15090, 21482, 71279, 12338, 65833, 13545, 19995, 24980,
       50270, 77124, 50800, 47239,  9718, 15878, 10077, 54761, 28330,
       40659, 28238,  6263, 16009, 73977, 21330, 57094,  8111, 45763,
       63092, 89226, 59345, 58848, 68197,  6876,   684,  2905, 47009,
       86580, 61114,  1881, 19558, 60506, 44160, 75266, 24414, 26311,
       28526, 23826, 84213, 16197, 55248, 77243, 54799, 87934,  5190,
       13984, 11744, 41008, 18902, 42154, 76979, 68881, 55126, 89685,
       63105, 20485, 41130, 75391, 61670, 14485,  7628, 24404, 64370,
       79144, 55225, 32043, 70205, 12615,   267, 16038, 44822, 26833,
       61649, 30474, 33135, 78851, 76872, 32276, 29887, 66517, 54117,
       58846, 88720,  1811,  9047, 82112, 63722, 89478, 44189, 68011,
       70632, 33667, 33596, 31713,   794, 60773, 53532, 23787, 54272,
       36712, 75851, 18985, 39835, 87786, 69302, 27391, 24670, 58914,
       58987,  3979, 17556, 56413, 79609, 56659, 40448, 30421, 30466,
       56373, 74982, 31931, 25909, 59698, 53576, 36429, 60652, 22014,
       37488, 59159, 84926, 80335, 84758, 12781, 79004, 77963, 36783,
       69367, 30493, 61960, 21791, 28916, 81640, 66368,  6947, 46538,
       38387, 61454,  2022, 70696, 81740, 80897, 85117, 61210, 51811,
       86599,  5516, 50547,  6881, 52349, 87263, 39301, 57079, 78244,
       36184, 63684, 31846, 35348, 35571, 79868, 84984, 44395, 22019,
       30356, 78940, 84326,  8516, 74426, 88432, 62503, 60505, 76077,
       49848, 71029, 56980, 59782, 47866,  1827, 36658, 51282, 49287,
       71651, 13304, 32591, 12536, 36898, 25893, 82528, 74749, 13338,
       85128, 57055, 16367, 55550, 88785, 62117, 40731, 70840, 78193,
       88179, 88306, 34930, 56998, 18756, 85261, 57614, 27586, 88966,
       53264, 82758, 57414, 57193, 68911, 67001, 56191, 40888, 39810,
       15820, 61209, 63071, 42012, 40951,  4840, 56011, 12378, 41537,
       74290,  8940, 79707, 89809, 21337,  5905, 43024, 43371,  8429,
        2214, 24827,  3112, 20890, 29505, 27847, 68686, 13697, 82526,
       78779, 69274,  1523, 38615, 20597, 62038, 51690, 21893, 72252,
       28983, 49626, 80758, 13895, 67944, 36395, 74094, 12243, 57408,
       19497, 23128, 14524, 31218, 65235, 27304, 35367, 41763, 66684,
       63876,  5884,  3078, 28939, 82168, 54903, 52159, 68126, 56355,
       18051, 32343, 43630,  4332, 84622, 69796, 10023, 85361,  7850,
       84691, 33402, 31000,  7724, 42887, 47897, 79155, 27318, 82673,
       29235, 14512, 33918, 60398, 67745, 67942, 27461, 11644, 51787,
       73610, 19438,  6644, 10886, 81957, 12458, 43642,  1738, 58210,
       69023, 63227, 73532, 74282, 38634, 69766, 87037, 79237, 24613,
       88183, 84659, 74812, 37530, 30618, 51538, 56936, 43356, 39688,
       65848, 39376,  6632, 20751, 13928, 55044, 58037, 31838, 56606,
       64669,   141, 44680, 22021, 49859, 64322, 41999, 26258, 75448,
        6090, 43886, 23161, 77057, 20529, 70641, 47777, 55369, 77052,
       74260, 62182, 62383, 43242, 53476, 23671, 17929, 79196, 80013,
       59954, 27897, 45277, 36398, 52385, 27584,  4933, 56243, 62592,
       44799, 58455, 18118, 68861, 67414, 26005, 44178, 79885, 35857,
       22552, 29404, 11111, 27288, 60669, 64814, 42829, 30823, 52329,
       12007, 21470, 87173, 63181, 74160, 54831, 60015, 48321,  2097,
       81874, 70705, 50672, 73116,   587, 61688, 48681,  5295, 47047,
       44361, 43632, 76864, 55644, 17420, 59377, 83967, 77205, 60251,
       16348, 41104, 39941, 52603, 52866, 76768,  8725, 21238,  1629,
       68925, 80821, 41349, 40836, 50826, 18983, 40852, 69845, 26205,
       12483,   938, 54359, 54529, 25539, 38180,  4053, 86527, 79771,
       78623, 64715, 12535, 83036, 20915, 24081, 73451, 54913, 50207,
       53260,  8565, 38218, 40965, 22710,  3702, 70895, 60765, 79317,
       13875, 17923, 84238, 22884, 87769,  1482, 20651, 74716, 17958,
       43897, 62867, 75725,  1685, 79728, 56281, 56804, 53627, 16338,
       77458, 72900, 53470, 77156, 59729, 34803,  9445,  1326,  3540,
       84119, 47908, 42002,   896, 27064, 26563, 43797, 45579, 75063,
       29745, 21805, 65242, 56989, 35408, 42339, 86260,   899, 78721,
       32459, 36708,   697, 45304, 43676, 20407, 30266, 20006, 37048,
       29347, 16805, 67146, 60833, 66555, 11427,  3863, 33962, 38909,
        6436, 37698,  9644, 29428,  5862, 45366, 84071, 32975, 84961,
       67709, 25208, 15858, 76571, 74415, 27098, 20146, 58118, 66065,
       70177, 33293, 16494,  7784, 54707, 51772,  3595, 67044, 65583,
       59213, 34406, 65564, 57845, 76512, 35671, 53958, 51941, 37061,
       58142, 32252, 72606, 18782, 25681,  4233, 87370, 74210, 28057,
        8936, 62144, 86417, 51539, 37819, 62511, 17565, 85429, 21824,
       47028, 12349, 34489, 59533,  8576, 24307, 74027,   516, 73591,
       84492, 46105, 69598, 19380, 54256, 67640, 22496, 88270, 78038,
        8198, 15206, 80021, 82512, 82149, 67676, 15937, 78079, 68238,
       38627, 69632, 48245, 52091, 72360, 38580, 52187,  6745, 78880,
       49428,  8709, 17925, 81328, 26523, 60231, 84729, 40100, 66941,
       43588, 71954, 31786, 37315, 28646,  7273, 43303,  5247,  2678,
       51051,  9479, 49221, 36321,  1899, 78051, 57797,  2263, 58614,
       47834, 83938, 71230, 14393, 86285, 21029, 67108, 23249, 54889,
       26518, 74295, 11091, 81264, 59069, 80470, 31621, 60487, 87088,
       87999, 10531, 79260, 20957, 74329, 87484, 45106, 17696, 68201,
       17461,  2629,  9159, 78049, 87578, 31524, 52713, 46780, 49719,
       55194, 18886, 65197, 48670, 19168, 22912, 50874, 87102, 85806,
       17310, 21111, 48517, 63093, 28075, 69301, 20166, 72324, 61326,
       56735,  6205, 49813, 54371, 65649, 65643, 89580,   872, 31108,
       22478, 61273, 12239, 53533, 60885, 14198, 10404,  4158, 86309,
       18473, 10462,  8236, 78024, 77669, 79186,  2556, 39355, 89967,
       70297, 63554, 24517, 86251, 30686, 21318, 55838, 46237, 61801,
       28997, 16397, 73212, 78825, 48765, 17812, 23524, 13611, 83002,
       18055,  9873, 85562, 76790,  8790, 38534, 11741, 38440, 87381,
       20592, 49000, 54583, 23134, 19243, 11268,   276, 87531, 48625,
       25947, 10152, 33639, 42413, 53741, 77012,  9538, 32565, 61463,
       86903, 71748,  1782, 43162, 40686, 73231, 16285, 25569, 15796,
       59168, 26348, 15927, 72072, 69342, 59592, 83881, 46885, 17277,
       48778, 67192,  8066, 42573, 78054, 42896, 54989, 22292, 83472,
       61470, 70284, 73846, 14438, 30429, 25194, 61838, 29896, 67868,
       56899, 61596, 54046,  9019, 56512, 21239, 30074, 13736, 71893,
       31849, 51750, 43745, 32916, 63094, 11238, 28985, 51083, 16808,
       66885, 60290, 40748, 57145, 88719, 44223,  9453, 79965, 61087,
       52415,   484, 88264, 20942, 50292, 70800, 44553, 57554, 13917,
       84457, 59181, 45319, 74165, 16693, 75794, 87379, 16543, 18744,
       43215, 28361,  5351,  7606, 14361, 89625, 55327, 64688, 23997,
       59041, 41009,  1041, 62686, 14499, 50658, 39844, 75646, 71939,
       63462, 84640, 15264, 68102, 79131, 51970, 80193, 40499, 83042,
       64092, 57488, 13452, 33016, 32719, 63774, 13310, 80268, 43976,
       34392, 40279, 76342, 41555, 14960, 17439, 69425, 18502, 45315,
       43933, 26932, 20873,  2219,  6045,  7782, 47724, 37239, 45115,
       36231, 42314, 58120, 27656, 57057, 30339, 85748, 77844, 67550,
       65839, 75181, 26229, 12985, 73440, 40708, 58638, 39187, 80226,
       37370, 57559, 68986, 69198, 33023, 52171, 23504, 30381, 19930,
       72587, 63990,   407,  9210, 72578, 51136, 66143, 48886, 64106,
       45481, 38056, 76045, 48195,  3563, 69923, 59672, 84363, 36920,
       20514, 16414, 80968, 38677, 11979, 62175, 34950, 20419, 81115,
       22201, 82649,  2313, 17205,  7963, 34108, 30101, 46571, 68343,
        4359, 19210, 23096,  7930, 31126, 68786, 73019,   455, 15299,
       13956, 62858, 85302, 82583, 66910,  6625, 26380, 39364, 20278,
       55049, 23483, 28388, 84728, 51486, 23721,  2631, 69095, 41612,
       43461, 30046, 76090, 45560,  6709, 19577, 68439, 26537, 67018,
       33381, 14061,   960,  6299, 66845, 44597,  9170, 78563, 30351,
       27838, 38372,  8290, 61612, 69024, 34237, 20476, 57986, 46868,
       53225, 79827, 56387, 71474, 56618, 65076, 20562, 35899, 71397,
       70236,  5078, 36311, 71264, 28258, 68273, 26946, 74133, 34586,
        9839, 35509, 20512, 20225, 21574, 21502, 62348, 64001, 45948,
       28514, 10343, 42860, 61370, 26154, 22370, 33560, 32797, 49036,
        8590, 10085,  6593, 13954,  4454, 85605, 17924, 87403, 16211,
       65642, 84907, 51033,  5220, 73510, 33584, 21232, 72460, 13357,
       20815, 53166, 45557, 84858, 10867, 13756, 36121, 21177, 81713,
       52014, 64062, 88838, 10236, 43314, 89187, 63708, 44340, 79465,
       27698, 88249, 85256, 70965, 65749, 16994, 38827, 23924, 23841,
       37078,  3991, 53348, 24126, 16206,  6451, 20234, 89853, 26194,
        7627, 53056, 11599,  2921, 24295, 60603, 43902, 16733, 19777,
       44375, 50027, 58127, 58843, 62973, 89266, 64185, 51897,  2111,
       76120, 45998, 42957, 51691, 34618, 70604, 74376, 64252, 10985,
       58715, 46605, 52013, 29327, 34032, 85309, 80751, 51768, 10324,
        8123, 60962, 38520, 78905, 54238, 51190, 47496, 54781, 43724,
       16094, 73056, 80871, 89155, 55688, 37572, 63683, 57335, 26527,
       80073, 48496, 50537, 43662, 41472, 36139, 89818, 29727, 45470,
       72695, 40937, 21413, 24776, 66440, 15948, 71722, 20842, 50025,
       70473, 78294, 20160, 16024, 63305, 10402, 55581, 34012,  5027,
       32770, 25621,  2372, 70934,  2548, 68257, 87900, 84682, 31318,
       13366, 45098, 46599, 48965, 25985, 52028, 78387, 63519, 35316,
       61625,  3647,  5590, 28973, 77771, 20405, 37458,  2291,  1756,
        2614, 63060,  4740, 72847, 62984, 51055, 36767,  5219, 10206,
       35902, 79459, 52128, 24082, 58078, 87854, 60347, 65152, 19842,
       74516, 45644,  5129, 52079, 34047,  5949, 36807, 32418, 31381,
       64317, 48089, 71982, 74152, 77842, 48644,  6504, 10487,  7751,
       21310,  8391,  4268, 69460, 20615, 33518, 86058, 79673, 21919,
       27496,  9094,  9861, 68029, 37680, 18678,  1165, 45808, 59743,
       10814, 72936,  3386, 57461, 82310, 69646, 77419,  4357,  7131,
       74253, 24399, 74682, 19927, 80531, 34752, 10644, 43002, 22024,
       75013, 28302, 13356, 30085, 37822, 72929, 82286, 87157, 24312,
       86277, 82122, 70015, 56639, 34233, 78879, 16150, 30998,  4170,
       31755, 44634, 80893, 68233, 46547, 86266, 13991, 83533, 83145,
       28156, 48250,  5178, 26629, 10230, 68141, 52416, 73476, 85222,
       29148, 29704, 14235, 24122, 55324, 14402, 75994, 71607, 16095,
        1270, 83729, 70051, 41461, 58231, 62540, 19872, 12610, 36161,
       58702, 30903, 52762, 49153, 12084, 28203, 39868, 47232, 55673,
       59763, 63466, 64544, 48637, 73735, 73323, 69663,  2028, 28048,
       47726, 82805, 58686, 28320, 24391, 70403, 19823, 36449, 19246,
        8096, 45606,  6740,   153, 61520, 74263, 26965, 35494, 68002,
       80930, 22954, 56229,  1084, 75721, 20711,  4399, 36909, 69453,
       54994, 37599, 72701, 50523, 53145, 67842, 15098, 88307, 67625,
       65584, 36223, 21596, 58055, 61213, 84997,  1310, 31153, 67876,
       65280, 85653, 83990, 89862, 48292, 17139, 34763, 47104, 73727,
       10662, 24691, 43928, 79314, 70295, 51900,  5663,  9009,  3265,
       69850, 67086, 40013,  6589, 68743, 11081, 70202, 83702, 58792,
       81188, 22804, 24177, 85540, 59196, 10001, 82015, 40411, 41820,
       40264, 56107, 37515, 63074, 46995,  1417, 64598,  8879, 76389,
       52330, 82774, 38302, 33460,  5626, 26553, 68541, 72617, 48004,
       25628, 38609,  4583,  6942, 57659, 88358, 79071, 52313, 19150,
       50460, 32335, 77405, 63443, 28097, 59830, 10643, 44086, 30958,
       75324, 84339, 42501, 56930, 25308, 83378, 55702,  9519, 71224,
       12417, 68092, 29632, 46374, 21585, 59926, 16299, 52785, 85221,
       17794, 76241, 36225, 33094, 87043, 17663, 81102, 53312, 43169,
       21589, 73817, 69155, 18351, 36269, 32856, 23750, 46401,   377,
       34270, 51279, 78882, 89239, 44691, 33772, 26625, 63832, 54568,
       87957, 72526, 68568, 11000, 30483, 65717, 12078, 19466, 10351,
       20399, 47520, 12485, 19095, 33203, 62366, 27754, 37996, 67520,
       41072, 13334, 31329, 47260,  2794, 64952, 28246, 79579, 26810,
       27366, 36181, 32721, 22264, 30002, 44963, 83496, 20126, 20238,
       25680, 41951, 25191, 26076, 88849,  8457, 37366, 24089, 49556,
       59148, 26033, 79163, 17875, 22239, 66669, 79124, 80894,  2530,
       45111, 78789, 73854, 62704, 32524, 82017, 40116, 13642, 71919,
       59434, 41867, 64050, 41056, 50834,  5187, 45746, 24725, 66131,
       47371, 68073, 89838, 16774, 81300, 87636, 31234, 70785,  8988,
       71000,  4921, 65079, 82913, 27165, 35149, 85032, 42728, 17101,
        2350, 57772, 83046, 73443, 13449, 73915, 64236, 65658,  5436,
       58299, 11146, 31738, 36640, 62871, 44755, 87791, 23623, 89990,
       57544, 48854, 56029, 13599, 25955, 25755, 18892,  5554, 82212,
       87082, 55432, 89287, 53939, 73737, 59573, 89417, 42789, 88509,
       61433, 23513, 74145, 54462, 48467, 41317, 56598, 25230, 56942,
       64211, 88656, 83058, 55184, 64002, 26223, 13171, 54416,  9396,
       12012,  3017, 50007, 47986, 70119, 41280, 53193, 33780, 89573,
       37174, 37828, 18736, 30165, 80856, 13632,  5186, 41994, 65379,
       14618, 60378, 27768, 14290, 39294, 32841, 44497, 50072, 52284,
       28077, 50605, 23653, 35116, 83276, 19849,  8272, 53489, 47870,
       82228, 30844, 65849, 49657, 30450, 51391, 56375, 51519, 74605,
       73374, 65419, 72723, 85547,  9130,  4731, 54214, 23694, 89250,
       33295, 58955, 32707, 59503,  3749, 17314, 27152, 14851, 35797,
       52505, 34477, 52114, 26513, 62109, 63308, 89316, 24803,  2484,
       22571, 50973, 51481,   417, 10641, 29311, 37979, 85146, 83765,
       18213,  9855, 21077,  8682, 47133, 83261, 39669, 38292, 10577,
       67136, 31664, 82708, 83237, 48166, 33814, 46612, 71836, 71723,
       60771, 19912, 51264, 68636, 17110, 84593, 67984,  2633, 20148,
       27516, 35223, 50318, 24188, 89440, 23670, 25315, 38648, 69887,
       17338, 76873, 38906, 29176, 79727,  9322, 16640, 86635, 50881,
       31293,  4916, 85909, 15790, 68270, 62428, 78898, 30928, 31158,
       60787, 14472,  2866, 31409, 32053, 47895, 79312, 25554, 89610,
       79469, 28240, 57833,  6075, 87451, 28096, 64220, 40953, 24846,
       50879, 43534, 74224,  7502, 18876, 11110, 72507, 62531, 72181,
       76706, 51626, 16950, 46861, 53401, 31050, 50174, 64654, 31637,
       30045, 17683, 60859, 63199, 53486, 49380, 58255, 53953, 49890,
       53800, 66759, 36966,  8875, 54708, 73356, 58103, 69498, 45332,
       71069, 30703, 43658, 74572, 69997, 45433, 67732, 69726, 51598,
       44670, 44218, 36958, 31463,  2081, 60462, 22181, 31584, 39173,
       38597, 51239, 50165, 19549, 33448, 14777, 43651, 16763, 52692,
       22070, 18434, 51786, 77821, 61374, 56441, 30011, 34778, 25362,
       16875, 65594,   941, 13788, 72094, 62106,  3823, 64109, 12826,
       26985, 27853, 42473, 24398, 62556, 34324, 82612, 11806, 50352,
       89612, 39336, 40737,  1962, 75938, 55855, 80246, 74758, 70503,
       77432, 42224, 85875, 86739, 82563, 87346, 86259, 66991, 60855,
       87646, 28956, 84375, 86975, 23940, 27508, 55165, 69250, 84784,
       82124, 47672, 42071, 27808, 74592,  3334, 89371, 33627, 83107,
       41980, 39328, 67472, 65575, 55818, 82641, 74491, 48948, 89506,
       44281, 19612, 46335, 25777, 32871, 43794, 36276, 59072, 65456,
       62865, 85539, 36265, 10978, 60327, 69567, 26519, 40201, 89726,
       68325, 75804, 39985, 23580, 28189, 44154, 64967, 50807, 65230,
       87621, 15342, 46386, 45041, 49317, 67767, 34394, 41053, 35946,
       40369, 44363, 64234, 49390, 27867, 76999, 37816, 48669, 15072,
       54952, 74483, 65585, 45970,  1065, 12019, 56014, 80823,  7217,
       44473, 25729, 66045, 56573, 66770, 40668, 43311, 34160, 33890,
       26599, 55270, 34048, 11200,  2713, 46161, 56576, 41076, 16216,
       50711,  5597, 30365, 31522, 15381, 77321, 40682, 83016, 24061,
       30342, 47036, 12868, 10179, 67824, 70269, 65510, 53067, 43787,
       51243,  6055,  5248, 22247, 64713, 32456, 73681, 24548,   832,
       75710, 75748,  4600, 10547, 65340, 47056, 52715, 87065, 47596,
       64875, 32991, 89821, 79666, 22336, 82418, 30455, 72644, 30263,
       10713, 76194, 61503, 35493,  8118, 14964, 56673,  1237, 54884,
       67670, 79391, 57620,  5323, 20712, 21121, 15755, 22656, 72052,
       20584, 58894, 18754, 89648, 59777, 28868, 51613, 15995, 25544,
       74732, 44986, 63664, 89020, 43458, 68158, 41329, 69100, 47651,
        8444, 31277, 83985, 15841, 34242, 80611, 43668, 42102, 65655,
       32902, 34949, 87834, 51254, 20804, 15207, 31688,  3803, 48505,
       53418, 84518, 51625, 21208, 78481, 14423, 66952, 51582, 50994,
       77568, 33362, 16784, 88589, 46799, 42676, 72463, 32232,  8011,
       85507, 12605, 12828, 29001, 31527, 36472, 28721, 22673, 76360,
       69015, 89157, 69256, 56805, 41920, 35720, 63503, 75621, 34984,
       81572, 36893, 85757,  9389, 82179, 78641, 21633,  9730, 68945,
       45993, 27387, 63792, 70512, 80656, 50944, 30519, 21734, 53412,
       41998, 68811, 12823, 66827, 54510, 26723, 28611, 68843, 85025,
       61995, 25606, 56028, 72935, 75299,  5021, 59793, 31079, 33319,
        4049, 39807, 79750, 33347, 86846, 61160,  3691, 29519, 29239,
       25949, 48329, 69351, 48501, 78115, 38267, 38612, 13349, 42200,
       17698, 67385, 35170, 28828,  6109, 60709, 89054, 51688, 38947,
       13827, 84025, 20409, 71021,  2964, 78934, 31222, 16000, 64567,
       49653, 58691, 19857, 80661, 39852, 43139, 14204, 10464, 45841,
       51820, 82000, 86655, 39348, 37122,  3473, 83711, 24349,  6406,
       67976, 24966,  7143, 61187, 31492, 30531, 23808, 77725, 47779,
        3052, 68793, 35933, 75431, 32357, 21340, 74794, 21073, 79526,
       22621, 18724, 58069, 83489, 58785, 39321, 13180, 56467,  7477,
         115, 17354,  1591, 73230, 77649, 79311, 52295, 12475, 50238,
       74039, 88195, 24275, 28224, 36461,  9461, 58735,  6563, 35417,
       29773, 59063, 35046, 70158, 27875, 39980,  9158, 72109, 18069,
       61628, 69470, 10036, 86070, 27956, 75028, 50642, 57138, 64708,
       73292, 80951,  6718, 23261, 88441, 22848,  7201, 51219, 34083,
       22596, 61231, 27091, 11628, 70552, 27258, 88603, 83537, 40935,
       65869, 39331, 11360, 34058, 36792, 28709, 67668, 29452,  2511,
       34454, 69488, 28535, 17178, 89098, 87914, 18090, 18774, 70037,
       59819,  9050, 72464, 82843, 72481, 88888, 41641, 61869,  2436,
       65870, 34035, 68296, 66024, 71189, 87598, 38533, 77363,  7509,
       89928,  7539, 34820, 19127, 62746, 74172, 46618,  2253, 23911,
       79966, 53521, 79444, 56349, 28618, 32094, 50116, 89279, 15473,
       15626, 55956,  3345, 82371, 73769, 28314, 12064, 85156, 63481,
       25925, 53836, 15403, 28250, 73959,  5704, 68345, 48251, 44424,
       44304, 39662, 82958,  1814,   532, 83868, 27046, 78669, 27694,
       47688, 47714, 42997, 17558, 28858, 40685, 12622, 50100, 14307,
         203, 65934, 52905, 27954, 16446, 58098, 10883, 33533, 32391,
       34024, 76239, 14271, 74932, 22625,  6028, 59574, 41651, 58457,
       68095, 64942, 40048, 46165, 37990, 17377, 53069, 43195, 10248,
       53757, 41979,  2440, 85923, 48197, 77716,  7236, 13840, 21564,
       25983, 22739, 57127, 38213, 81474,  5117, 62938, 44398, 49636,
       30890, 42336, 74551,  9281,  8059, 59866, 74697, 67072, 32413,
       81617, 38592, 70485, 23996, 62467, 58289, 85277, 55859, 72966,
       58420, 13950,  9540, 67058, 37763, 18735, 71131, 38945, 34685,
       66344, 78212, 47382, 44695, 61945, 40547, 66075, 75637,  1651,
       36463, 52657, 17066, 65971, 47701, 85538, 14199, 68182, 76406,
       66323,  2231, 70708, 48585, 67450, 62816, 37855, 67646, 16970,
       71772, 72593, 77044, 78878, 39172, 31945, 86936, 49842, 87603,
       17289, 29657, 10731, 72533, 10709, 48028, 71543, 66822, 79266,
       33771, 67519, 45641, 70689, 87487,  7777, 32895, 73844, 79739,
       61735, 34173, 34493,  2767,  9515, 71340, 35088, 74026, 60786,
       67205,  9390, 89223, 53247, 75797, 40252, 35275,  2695, 29333,
       26165, 60048, 75485, 57503, 56720, 70303, 57945, 72254, 33072,
       51474, 46086, 38199, 63834, 24916, 32941, 70526, 89225,  6598,
       77267, 26315, 77101, 28964, 43893, 41471,  1058, 52265, 66200,
       14413, 48543, 42405, 46319, 68914,  5271, 49806, 85607, 47742,
       50241, 67437, 76349, 62164,   608,  3010, 50582, 18556, 64750,
       42849, 17986, 39953, 55635,  6352,  6161,  4120, 49700, 30822,
       65793, 11941, 17674, 52134,  4808, 88781, 34793, 72691, 31859,
       49863, 59973, 30653, 10371,  7320, 77969, 47607, 26814, 30480,
        9027,  1438,  9270, 47846, 71720, 14213, 18383, 87360, 22112,
       33728, 10862, 26704, 72238, 57411, 79742, 70229, 12379, 86808,
       53485, 35583, 85082, 16540, 31844, 80924, 65303,  4377, 56300,
       73651,  6945, 58650, 51526, 41520, 72436, 12994, 47964,  4045,
        7583, 28407,  4019,   330, 68806, 13938, 70517, 33567, 50386,
        7032, 22814, 21712, 50678, 66993, 74886, 89389, 31880, 63058,
       16310, 27907, 76901, 43029, 51832, 36695, 37165, 58620,   801,
       64682, 12367, 88500, 78906, 71798, 26757, 64187, 89274,  4785,
       63752, 36153, 24990, 31423, 34659,  2115, 89912,  7380, 83671,
       32371, 25334, 30032, 81064, 51095, 48440, 45529, 15653, 58739,
       70830, 38001,  8991, 22446, 78542, 55609, 84547, 54571, 24756,
       62305, 79145, 13213, 47899, 18627, 19913, 86625, 15778, 73299,
       34034, 50349, 27836,  1037, 85749, 17627, 65797, 32958, 66306,
       77235, 35944, 65544, 89787,  4551,  4037, 25155, 45801, 71527,
        1170,  1958,  3731, 70812, 89426, 48990, 62700, 51910, 28665,
       49414, 35155, 15898, 21642, 42950, 47653, 29335,  1627, 87868,
       71594, 50753, 53110, 80299, 35760, 17491,  3286, 88492, 35300,
       83753, 64460, 66418, 40730, 25344, 42650, 32823, 85069, 38049,
       19580, 73564, 70140, 61841, 14821, 59043, 56876, 50551, 34342,
        6685, 42913,  9627, 37592, 14160, 52436, 58953, 35082, 38169,
       63135, 26584, 17489, 87867, 48892, 68899, 19654, 10213,  5830,
       56340, 63860, 46886, 34326, 21876, 62019, 53130, 45059, 70727,
       19860, 51957, 26401, 11957, 88947, 77334, 64861, 78923, 58675,
       88992, 28859, 30203, 40043, 41336, 18504, 66978, 81838, 32615,
       80414, 37033, 45698, 45730, 24632,  7988, 87841, 54176, 85070,
       47543,    58, 29600, 12780, 78644, 10741, 46699, 89861, 27135,
       34938, 79257, 67306, 23296, 76008, 21895, 40286, 22147, 32360,
       48664, 11783, 78129, 71685, 80243, 50250, 37605, 35101, 82451,
       21260, 39005, 19262, 33320, 26644, 32865, 22501, 69004, 55949,
       83722, 70943, 31264,  9195, 23105,  5414, 62309, 20555, 70798,
       89747, 80439,  5515, 75981, 76208, 25222, 35038, 20845, 55684,
       51742, 75158, 83486, 69258, 38380,  2090, 25432, 82606, 79806,
        5803, 43116, 31942, 35717, 74582, 74917, 70868, 89253, 30166,
       77890, 57248, 77398, 34305, 69436,  6637, 35888, 83469, 29627,
       38316, 76817, 15159, 30842, 36845,   810,  3243, 24511, 48117,
       73459, 21106, 34442, 39979, 39465, 51560, 71003, 13167, 39748,
       43925, 44883, 62412, 44908, 75614, 69430, 10702,  5080, 82026,
       48451, 26855, 48406, 66922, 61218, 17130, 54082, 67659,  5855,
       39576,  6309, 60649, 76402, 21683, 60610, 61297, 19487, 55589,
       45565, 60470, 88729, 16824, 68003,  9288, 88487, 58876, 20896,
       17601, 36247, 84471, 73355, 67754, 36405, 31636, 33437, 57170,
       86650, 71050, 88131, 16611, 80736, 59745, 80483, 35018, 42521,
       76616, 45601, 12513, 84488, 14270, 89276, 46938, 27185,  7569,
       44909, 80966, 49340, 69560, 58856, 26125, 84161, 62908, 27384,
       48822,  7699, 80953, 15191, 87098, 77286,  4643, 64739, 11656,
       46728, 74507, 45166,  9534, 59453, 50310, 35216, 76580, 80475,
       40412, 64409, 24554, 23356, 89300, 37425, 18299, 32466, 72974,
       48939, 18638, 58152, 17351, 22208, 15463, 39320, 64401, 12036,
       62036, 52679, 88758, 10441, 26096, 25472, 68316, 73102, 10605,
       16552, 31091, 73079, 47864, 63518, 73588, 50591, 65788, 68090,
       87984, 34946, 43581,  8334, 86663, 58023, 86886, 78027, 35685,
       43367, 10456, 82963,  5004, 61797, 58108, 37619, 68970, 49972,
       70207, 61146, 64349, 14398, 22310, 69382, 50747, 43506, 63484,
       32751,  6243, 70982, 70577, 73704,  8965, 74654, 37578, 77996,
       52455, 64788, 40080, 28185, 33175,  7516, 58497, 68144, 89351,
       42581, 42306, 28364, 10725, 46285, 84645,  8883, 69088, 26366,
       77406, 24437, 64188, 10091, 80451, 32945, 22590, 27594, 10849,
         444, 19102, 88149, 12225, 59176, 38955, 52503, 48430,  6584,
        5354, 71082, 31748, 39447, 38272, 56183, 55155,  3896, 46409,
       73119, 76038, 26763, 44855, 49795, 24665, 21065, 82690, 70120,
       68894, 28489, 17786, 50502,  2401, 84796,  8121, 51945, 19458,
       84569, 15537, 24641, 26594, 67887, 61398,  8106, 55512, 66261,
       60268, 36914, 20411, 56722, 86540, 58495, 52748, 67406, 83977,
       83208, 54829, 36047, 50080, 28895, 76555,  6328, 14191, 55783,
       47071, 87947, 89673, 83245, 70983, 53443, 87442, 13789, 83305,
       62373, 68420, 12753, 33266, 80952, 54627, 59435, 76486, 82130,
       50763, 17167, 47865, 46129, 85381, 14799, 75852, 87685, 25103,
       52533, 52129, 61524, 87138, 27929, 55354,  3968, 25393, 42230,
       33803, 21966, 36965, 79661, 47363, 21312, 32283, 10616, 72664,
       77288, 21956, 72851, 68419, 84472, 85066, 20109, 24033, 12322,
       27281, 16372, 52212, 46089, 77519, 17103, 21516, 43802, 58510,
       57546,  9818, 17369, 53792,  8500, 40614, 61613, 44039,  3618,
       24741, 81564, 79504, 23367, 21978, 86674, 40003, 23095, 14658,
         725, 59949, 62374, 69856,  5372, 20779, 46132, 73484, 41725,
       17658, 14482, 68787, 48639, 34896, 16983, 66256, 65533, 63141,
        2304, 14317, 52939, 89923, 12928, 60572, 62919,  9782, 83494,
       26085, 10329, 74566, 42271, 38081, 26372, 89780, 52860, 66335,
       65436, 80517, 17932,  8891, 31116, 66842,  5940, 17446, 12383,
       72016, 77690, 69417, 18137, 16546, 34897, 86258, 59411, 21703,
       41197, 44145, 68281, 56297, 53033,  1622, 67248, 68113, 27774,
       59136,  1338, 85827, 19333,  2580, 80760, 67576, 88474, 60193,
       89257, 16619, 16395, 40841, 22380, 78452, 49130, 30406, 22329,
       45063, 82477, 87384, 27900, 65133, 16205,  9659, 29803, 57774,
       43421,  5208, 12109, 86102, 32269, 19647, 11684, 88718, 46315,
       17316, 83714, 21015, 81726, 13315, 66935, 52310, 31388, 68845,
       71563, 47509, 76483,   548, 73635, 73478, 34750, 81901, 78208,
       89238, 24713, 68070, 11085,  3365,  6366, 64891, 86386, 36937,
       77270, 54286,  4846, 82007, 60562, 87266, 62211, 47953, 86460,
       62579,   233,  6846, 16656, 56113, 74519, 81518, 22735, 70700,
       56638, 19324, 60413, 68237, 66604, 20625, 11206, 68680, 64414,
        7855, 38197, 60638, 74075,  3904,  1981, 12376, 38527, 12796,
       23389, 13381, 10697, 23985, 39857, 87416, 27868, 23950, 10312,
       10084, 36890, 62784, 15786, 24034, 27232, 34274, 47906, 78197,
       44489, 62527, 44190, 31189, 68211,  1329,  1352, 25654, 72892,
       81254, 14807, 68396,  6279, 28457, 17120, 40591, 53803, 27134,
       34645, 19590, 71645, 34606,    89, 30159, 63715,  7960, 54370,
       84865, 15835, 22569, 80053, 84279, 79968, 16537, 83207, 17801,
       52582, 22649, 41678, 52278, 20363, 44872, 56209, 25037, 56526,
       48243, 71899, 10405, 48468, 16117, 43801, 88097, 32610, 25847,
       72763, 29725, 52996, 11095, 70498, 26856, 46800,  4471, 17085,
       72131, 30355, 84002,  3872, 24417, 64141, 31745, 40938, 21212,
       86799, 76383, 89361, 45978,  9393,   467, 59821, 48626, 33123,
       35542, 54786, 48563, 31909, 13083, 84966, 51791, 76067, 11658,
       38547, 46011,  6866, 31155, 51185, 27470, 37182, 18121, 57820,
        8814, 73785, 74995, 85288, 25521,   501,  4043, 38038, 13025,
       82411, 48053, 73755, 74819, 48010, 71825,  6429, 24599,  5888,
       33292, 83229, 29885, 56520, 17353, 27921, 68030, 51763, 46362,
        9398, 77386, 48827, 89801, 54813, 41312, 30774, 22279, 70332,
       68723, 10388, 45915, 79564, 31445, 60692, 12968, 27663, 89505,
       11403, 15281, 66562, 52406, 33158, 12926, 24019,  3333, 70054,
       63868, 66961, 80392, 16839, 10700, 68956, 50945, 51866, 68265,
       53507, 79132, 58235, 64742, 69791, 13852, 34690, 26811, 34867,
       60978,  6174, 45595, 36763, 67725, 31007, 81161, 57843, 44641,
       82074, 45186, 36604, 57045, 51677, 26091, 15353, 21282, 36886,
       41885, 79589, 72058, 70385, 68885, 28180, 48328, 86946, 84585,
        3516, 84385, 81274, 29688, 12618, 72525, 47658, 40797,  4132,
       19139, 35158, 76278, 76544, 36837, 48094, 32888, 72854, 11478,
       64711, 46157, 28874,  4402, 16304, 43839, 79435, 79958, 18663,
       32600,  1111, 43387, 89663, 17222, 34354, 75595, 28242, 88356,
       66376, 36298, 67039, 54495, 64037, 13385,  4258, 78160, 24405,
       86060,   280, 19176, 56567, 73598, 35442, 61343, 53395, 86866,
       39289, 54135, 49992, 49619, 59016])