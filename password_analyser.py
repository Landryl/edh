import matplotlib.pyplot as plt
import numpy as np
import time

def read_int(f):
    ba = bytearray(4)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.int32)
    return prm[0]
    
def read_double(f):
    ba = bytearray(8)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.double)
    return prm[0]

def read_double_tab(f, n):
    ba = bytearray(8*n)
    nr = f.readinto(ba)
    if nr != len(ba):
        return []
    else:
        prm = np.frombuffer(ba, dtype=np.double)
        return prm
    
def get_pics_from_file(filename):
    # Lecture du fichier d'infos + pics detectes (post-processing KeyFinder)
    print("Ouverture du fichier de pics "+filename)
    f_pic = open(filename, "rb")
    info = dict()
    info["nb_pics"] = read_int(f_pic)
    print("Nb pics par trame: " + str(info["nb_pics"]))
    info["freq_sampling_khz"] = read_double(f_pic)
    print("Frequence d'echantillonnage: " + str(info["freq_sampling_khz"]) + " kHz")
    info["freq_trame_hz"] = read_double(f_pic)
    print("Frequence trame: " + str(info["freq_trame_hz"]) + " Hz")
    info["freq_pic_khz"] = read_double(f_pic)
    print("Frequence pic: " + str(info["freq_pic_khz"]) + " kHz")
    info["norm_fact"] = read_double(f_pic)
    print("Facteur de normalisation: " + str(info["norm_fact"]))
    tab_pics = []
    pics = read_double_tab(f_pic, info["nb_pics"])
    nb_trames = 1
    while len(pics) > 0:
        nb_trames = nb_trames+1
        tab_pics.append(pics)
        pics = read_double_tab(f_pic, info["nb_pics"])
    print("Nb trames: " + str(nb_trames))
    f_pic.close()
    return tab_pics, info
    
pics_nokey, info = get_pics_from_file("/content/data/pics_NOKEY.bin")
pics_pad0, info = get_pics_from_file("/content/data/pics_0.bin")

from os import listdir
from os.path import isfile, join, splitext

binpath = "/content/data/"
outPath = "/content/out_data/"
onlyfiles = [f for f in listdir(binpath) if isfile(join(binpath, f))]
for f in onlyfiles:
    print("Starting treatment of " + f)
    (pics, info) = get_pics_from_file(binpath + f)
    outfileName = outPath + splitext(f)[0] + ".csv"
    outfile = open(outfileName, "w+")
    for pic in pics:
        for p in range(len(pic) - 1):
            outfile.write(str(pic[p]) + ",")
        outfile.write(str(pic[len(pic) - 1]) + "\n")
    outfile.close()
    print("Done. Data stored in " + outfileName + "\n\n")
    
from os import listdir
#load every data
keys = []
dataset_full = None
dataset_correspondences = None
for file_name in listdir('/content/out_data'):
  key = file_name[5:len(file_name) - 4]
  if (key != "LOGINMDP"):
    keys.append(key)
    print(keys)
    #new datas
    dataset = loadtxt('/content/out_data/' + file_name, delimiter=',')
    if (dataset_full is None):
      dataset_full = dataset
    else:
      dataset_full = numpy.concatenate((dataset_full, dataset))
    
    #correspondences
    zero_one = [0] * 42 #total nbs of keys
    zero_one[len(keys) - 1] = 1
    matrix_zero_one = numpy.array([zero_one] * len(dataset))
    if (dataset_correspondences is None):
      dataset_correspondences = matrix_zero_one
    else:
      dataset_correspondences = numpy.concatenate((dataset_correspondences, matrix_zero_one))
    print(dataset_correspondences)
    print(dataset_full)
    
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X, y = shuffle(dataset_full, dataset_correspondences)
X_train, X_test, y_train, y_test = train_test_split(dataset_full, dataset_correspondences, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# priting confusion matrix

from sklearn.metrics import confusion_matrix
import itertools

rounded_y_test = np.argmax(y_test, axis=1)
rounded_y_pred = np.argmax(y_pred, axis=1)

#cm = confusion_matrix(rounded_y_test, rounded_y_pred)

#plt.figure(figsize=(14,14))
#np.set_printoptions(precision=2)

#plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title("Confusion Matrix")
#plt.colorbar()
#tick_marks = np.arange(len(keys))
#plt.xticks(tick_marks, keys, rotation=45)
#plt.yticks(tick_marks, keys)

#fmt = 'd'
#thresh = cm.max() / 2.
#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, format(cm[i, j], fmt),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")

#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.tight_layout()
#plt.show()

dataset_loginmdp = loadtxt('/content/out_data/pics_LOGINMDP.csv', delimiter=',')
prediction = regressor.predict(dataset_loginmdp)

#for i in range(42) :
#  plt.figure(1)
#  plt.title(keys[i])
#  plt.plot(range(len(prediction[:,i])), prediction[:,i], 'k')
#  plt.show()

treshold = 0.5

password_probabilites = ['']

for n in range(len(prediction)):
  # NOKEY, d'après notre matrice de confusion, est fiable
  # On va donc filtrer les trames intéréssantes avec
  rounded_trame_prediction = np.argmax(prediction[n])
  if keys[rounded_trame_prediction] != "NOKEY":
    probable_key = ''
    for i in range(42):
      if keys[rounded_trame_prediction] == "SHIFT":
        continue
      if prediction[n][i] > treshold:
        probable_key = keys[i]
    if probable_key != '' and password_probabilites[-1] != probable_key:
      password_probabilites.append(probable_key)

print(password_probabilites)

prev = ''
for k in password_probabilites:
  if k != 'CTRL' and k != 'SUPPR' and k != 'SHIFT' and k != 'ENTER':
    if k != prev:
      print(k, end='')
      prev = k
