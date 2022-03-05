from numpy import loadtxt
import keras

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
model= keras.models.load_model("model.h5")

predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(15):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))