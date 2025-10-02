import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd
import ROOT as r
from array import array

########################################## CONSTANTS ############################################
cwd = os.getcwd()
img_dir = f'{cwd}/../boostrapWorkflow/imagesKDE/'
gan_img_dir = f'{cwd}/../GAN_workflow/GAN_bootstrapSamples_workflow/imagesKDE/'
MODEL_WEIGHTS = f'{cwd}/../training_bootstrap/model_KDE_2.1.h5'
#MODEL_WEIGHTS = f'{cwd}/../training_bootstrap/model_4.0.keras'
LR = 0.00005
activ='relu'
######################################## END CONSTANTS ##########################################

#lo pones a false para que solo te coja las capas de extracción de características y no las densas
base_model = ResNet50(weights='imagenet', 
                      include_top=False,
                      input_shape=(796,772,3))
# Esto lo puedes modificar en función de si quieres o no que se modifiquen los pesos durante el ajuste fino
for layer in base_model.layers: layer.trainable = False
# build model ########################################################################
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024, activation=activ))
model.add(Dense(1024, activation=activ))
#model.add(Dropout(0.05))
model.add(Dense(512, activation=activ))
model.add(Dense(512, activation=activ))
model.add(Dense(256, activation=activ))
#model.add(Dropout(0.05))
model.add(Dense(1, activation='linear'))
model.summary()

lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
    LR,
    decay_steps=100,
    decay_rate=0.9,
    staircase=True)
model.compile(optimizer=Adam(learning_rate=lr_sched), loss='mean_absolute_percentage_error')
model.load_weights(MODEL_WEIGHTS)
#######################################################################################

print("GPUs Available: ", str(tf.config.list_physical_devices('GPU')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# GPU memory usage configuration

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

radius = {
    "_20mm": 20.,
    "_18mm": 18.,
    "_16mm": 16.,
    "_14mm": 14.,
    "_12mm": 12.,
    "_10mm": 10.,
    "_8mm": 8.,
    "_6mm": 6.,
    "_4mm": 4.,
}

test_datagen = ImageDataGenerator(rescale=1./255)

# build dataframe
data = []
for f in os.listdir(img_dir):
    if not f.endswith('.png'): continue
    for key in radius:
        if key in f: t = radius[key]
    whatfor = 'train'
    if any(s in f for s in [f'_{i}_' for i in range(0,10)]): 
        whatfor = 'test'
    else: continue
    data.append((f,t,whatfor))
for f in os.listdir(gan_img_dir):
    if not f.endswith('.png'): continue
    for key in radius:
        if key in f: t = radius[key]
    if any(s in f for s in [f'_{i}_' for i in range(1000,1010)]): 
        whatfor = 'test_gan'
    data.append((f,t,whatfor))

df = pd.DataFrame(data, columns=['filename', 'thickness', 'whatfor'])
print(df)

test_generator = test_datagen.flow_from_dataframe(df.loc[df['whatfor']=='test'],
                                                  x_col='filename',
                                                  y_col='thickness',
                                                  directory=img_dir,
                                                  target_size=(796, 772),
                                                  batch_size=1, 
                                                  class_mode=None, 
                                                  shuffle=False)
test_gan_generator = test_datagen.flow_from_dataframe(df.loc[df['whatfor']=='test_gan'],
                                                  x_col='filename',
                                                  y_col='thickness',
                                                  directory=gan_img_dir,
                                                  target_size=(796, 772),
                                                  batch_size=1, 
                                                  class_mode=None, 
                                                  shuffle=False)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
STEP_SIZE_TEST_GAN=test_gan_generator.n//test_gan_generator.batch_size

with tf.device('/GPU:0'):
 
    test_generator.reset()
    predictions=model.predict(test_generator,
                       steps=STEP_SIZE_TEST,
                       verbose=1)

    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions.flatten(),
                          "True values":df.loc[df['whatfor']=='test'].thickness.tolist()})
    print(results)
    predictions_gan=model.predict(test_gan_generator,
                       steps=STEP_SIZE_TEST_GAN,
                       verbose=1)

    filenames=test_gan_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions_gan.flatten(),
                          "True values":df.loc[df['whatfor']=='test_gan'].thickness.tolist()})
    print(results)

# Make plot
line = r.TGraph(2, np.array([0.,24.]), np.array([0.,24.]))

x_g4 = df.loc[df['whatfor']=='test'].thickness.tolist()
y_g4 = predictions.flatten()
tprof_g4 = r.TProfile("tprof_g4", ";Real thickness (mm);CNN-predicted thickness (mm)", 22, np.array([i-0.5 for i in range(23)], dtype=float), 0., 24.)

x_gan = df.loc[df['whatfor']=='test_gan'].thickness.tolist()
y_gan = predictions_gan.flatten()
tprof_gan = r.TProfile("tprof_gan", ";Real thickness (mm);CNN-predicted thickness (mm)", 22, np.array([i-0.5 for i in range(23)], dtype=float), 0., 24.)

print('- GEANT4 results:')
for i in range(len(x_g4)):
    print(x_g4[i], y_g4[i])
    tprof_g4.Fill(x_g4[i], y_g4[i])
dev = np.mean(y_g4-x_g4)
stddev = np.std(y_g4-x_g4)
print(f'  Deviation from real values: {dev} +- {stddev}') 

print('- cGAN results:')
for i in range(len(x_gan)):
    print(x_gan[i], y_gan[i])
    tprof_gan.Fill(x_gan[i], y_gan[i])
dev = np.mean(y_gan-x_gan)
stddev = np.std(y_gan-x_gan)
print(f'  Deviation from real values: {dev} +- {stddev}') 

graph_g4 = r.TGraph(len(x_g4), np.array(x_g4, dtype=float), np.array(y_g4, dtype=float))
graph_gan = r.TGraph(len(x_gan), np.array(x_gan, dtype=float), np.array(y_gan, dtype=float))

r.gROOT.ProcessLine('.L ./tdrstyle.C')
r.gROOT.SetBatch(1)
r.setTDRStyle()
r.gStyle.SetErrorX(0.)

c = r.TCanvas("c", "c", 800, 800)
c.cd()
line.GetXaxis().SetTitle("Real thickness (mm)")
line.GetYaxis().SetTitle("CNN-predicted thickness (mm)")
line.GetXaxis().SetRangeUser(0.,24.)
line.SetMaximum(24.)
line.SetMinimum(0.)
line.SetLineStyle(2)
line.SetLineWidth(2)
line.SetLineColor(r.kBlack)
line.Draw("AL")

tprof_g4.SetMarkerStyle(20)
tprof_g4.SetMarkerSize(1)
tprof_g4.SetMarkerColor(r.kRed+1)
tprof_g4.SetLineColor(r.kRed+1)

tprof_gan.SetMarkerStyle(20)
tprof_gan.SetMarkerSize(1)
tprof_gan.SetMarkerColor(r.kBlue+1)
tprof_gan.SetLineColor(r.kBlue+1)

graph_g4.SetMarkerStyle(24)
graph_g4.SetMarkerSize(0.4)
graph_g4.SetMarkerColor(r.kRed-7)

graph_gan.SetMarkerStyle(24)
graph_gan.SetMarkerSize(0.4)
graph_gan.SetMarkerColor(r.kBlue-7)

#graph_g4.Draw("P,SAME")
#graph_gan.Draw("P,SAME")
tprof_g4.Draw("PE1,SAME")
tprof_gan.Draw("PE1,SAME")

l = r.TLegend(0.2,0.7,0.4,0.83)
l.AddEntry(tprof_g4, "GEANT4-simulated images", "P")
l.AddEntry(tprof_gan, "cGAN-generated images", "P")
l.SetFillStyle(0)
l.SetTextFont(42)
l.SetTextSize(0.025)
l.SetBorderSize(0)
l.Draw()
c.SaveAs(f"CNN_plot_profile_final_KDE_2.1.png")
c.SaveAs(f"CNN_plot_profile_final_KDE_2.1.pdf")
