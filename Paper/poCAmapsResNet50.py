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

########################################## CONSTANTS ############################################
img_dir = './PoCAmaps/'
TRAIN = False
EPOCHS = 50
LR = 0.00005
activ='relu'
BATCH_SIZE=8
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
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation=activ))
model.add(Dropout(0.1))
model.add(Dense(512, activation=activ))
model.add(Dropout(0.1))
model.add(Dense(256, activation=activ))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(optimizer=Adam(lr=LR, decay=LR/200), loss='mean_absolute_percentage_error')
model.load_weights('./training_3/model.keras')
#######################################################################################

print("GPUs Available: ", str(tf.config.list_physical_devices('GPU')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# GPU memory usage configuration

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


radius = {
    "18p0": 20.,
    "18p2": 18.,
    "18p4": 16.,
    "18p6": 14.,
    "18p8": 12.,
    "19p0": 10.,
    "19p2": 8.,
    "19p4": 6.,
    "19p6": 4.,
    "19p8": 2.,
}

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# build dataframe
data = []
for f in os.listdir(img_dir):
    if not f.endswith('.png'): continue
    for key in radius:
        if key in f: t = radius[key]
    whatfor = 'train'
    if 'seed20' in f or 'seed19' in f: whatfor = 'test'
    if 'seed18' in f or 'seed17' in f: whatfor = 'validation'
    if 'gan' in f: whatfor = 'test_gan'
    data.append((f,t,whatfor))

df = pd.DataFrame(data, columns=['filename', 'thickness', 'whatfor'])
print(df)

train_generator = train_datagen.flow_from_dataframe(df.loc[df['whatfor']=='train'],
                                                    x_col='filename',
                                                    y_col='thickness',
                                                    directory=img_dir,
                                                    target_size=(796, 772), 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='raw')
valid_generator = valid_datagen.flow_from_dataframe(df.loc[df['whatfor']=='validation'],
                                                    x_col='filename',
                                                    y_col='thickness',
                                                    directory=img_dir, 
                                                    target_size=(796, 772),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='raw')
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
                                                  directory=img_dir,
                                                  target_size=(796, 772),
                                                  batch_size=1, 
                                                  class_mode=None, 
                                                  shuffle=False)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
print(f'STEP_SIZE_TRAIN: {STEP_SIZE_TRAIN}, STEP_SIZE_VALID: {STEP_SIZE_VALID}, STEP_SIZE_TEST: {STEP_SIZE_TEST}')


with tf.device('/GPU:0'):

    if TRAIN:        
        model.fit(train_generator, 
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=valid_generator,
              validation_steps=STEP_SIZE_VALID,
              epochs=EPOCHS)
        model.save('./training_3/model.keras')

    model.evaluate(valid_generator,
                   steps=STEP_SIZE_VALID)

 
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
                       steps=STEP_SIZE_TEST,
                       verbose=1)

    filenames=test_gan_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions_gan.flatten(),
                          "True values":df.loc[df['whatfor']=='test_gan'].thickness.tolist()})
    print(results)

# Make plot
line = r.TGraph(2, np.array([0.,24.]), np.array([0.,24.]))

x = df.loc[df['whatfor']=='test'].thickness.tolist()
y = predictions.flatten()
print(x)
print(y)
graph_g4 = r.TGraph(len(x), np.array(x, dtype=float), np.array(y, dtype=float))
x = df.loc[df['whatfor']=='test_gan'].thickness.tolist()
y = predictions_gan.flatten()
graph_gan = r.TGraph(len(x), np.array(x, dtype=float), np.array(y, dtype=float))

r.gROOT.ProcessLine('.L ./tdrstyle.C')
r.gROOT.SetBatch(1)
r.setTDRStyle()

c = r.TCanvas("c", "c", 800, 800)
c.cd()
line.GetXaxis().SetTitle("Real thickness (mm)")
line.GetYaxis().SetTitle("CNN-predicted thickness (mm)")
line.SetMaximum(24.)
line.SetMinimum(0.)
line.SetLineStyle(2)
line.SetLineWidth(2)
line.SetLineColor(r.kBlack)
line.Draw("AL")

graph_g4.SetMarkerStyle(20)
graph_g4.SetMarkerColor(r.kRed+1)
graph_g4.Draw("P,SAME")

graph_gan.SetMarkerStyle(20)
graph_gan.SetMarkerColor(r.kBlue+1)
graph_gan.Draw("P,SAME")

l = r.TLegend(0.2,0.7,0.4,0.83)
l.AddEntry(graph_g4, "Geant4 test images", "P")
l.AddEntry(graph_gan, "GAN-generated images", "P")
l.SetFillStyle(0)
l.SetTextFont(42)
l.SetTextSize(0.025)
l.SetBorderSize(0)
l.Draw()
c.SaveAs("CNN_plot.png")
