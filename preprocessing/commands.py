import os

## First 50 Subjects
# subjects  = [100610, 102311, 102816, 104416, 105923,
#              108323, 109123, 111514, 114823, 115017,
#              115825, 116726, 118225, 125525, 126426,
#              128935, 130114, 130518, 131217, 131722,
#              132118, 134627, 134829, 135124, 137128,
#              140117, 144226, 145834, 146129, 146432,
#              146735, 146937, 148133, 150423, 155938,
#              156334, 157336, 158035, 158136, 159239,
#              162935, 164131, 164636, 165436, 167036,
#              167440, 169040, 169343, 169444, 169747]

subjects  = [185442, 186949, 187345, 191033, 191336,
             191841, 192439, 192641, 193845, 195041]

for sub in subjects:
    print("Start processing " + str(sub))
    command = "python3 pre.py " + str(sub)
    os.system(command)
    print("success " + str(sub))
