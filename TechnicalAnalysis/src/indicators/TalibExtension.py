import numpy as np;
import talib as ta;

def KDJ(high, low, close, kPeriods=14, dPeriods=3):
	vectorSize = high.shape[0];
	fastK = np.zeros(vectorSize);
	fastK[:] = np.NAN;
	lowestLowVector = np.zeros(vectorSize);
	lowestLowVector[0:kPeriods] = min(low[0:kPeriods]);
	for i in range((kPeriods-1), vectorSize):
		lowestLowVector[i] = min(low[(i-kPeriods+1):i+1]);
	highestHighVector = np.zeros(vectorSize);
	highestHighVector[0:kPeriods] = max(high[0:kPeriods]);
	for i in range(kPeriods-1, vectorSize):
		highestHighVector[i] = max(high[(i-kPeriods+1):i+1]);
	highLowDiff = highestHighVector - lowestLowVector;
	nonZero = highLowDiff.ravel().nonzero();
	fastK[nonZero] = np.divide((close[nonZero] - lowestLowVector[nonZero]),
							   (highestHighVector[nonZero]-lowestLowVector[nonZero])) * 100;
	fastD = np.zeros(vectorSize);
	fastD[:] = np.NAN;
	fastD[~np.isnan(fastK)] = ta.EMA(fastK[~np.isnan(fastK)], dPeriods);
	jLine = 3*fastK-2*fastD;
	
	return np.hstack((fastK.reshape((fastK.size, 1)), fastD.reshape((fastD.size, 1)),
					 jLine.reshape((jLine.size, 1))))
	
def TSI(close, slowPeriod=25, fastPeriod=13):
	if ((fastPeriod >= close.size) | (slowPeriod >= close.size)):
		return;
	momentumVector = np.zeros(close.size);
	momentumVector[1:] = close[1:] - close[0:(close.size-1)]
	absMomentumVector = np.abs(momentumVector);
	k1 = 2/(slowPeriod+1);
	k2 = 2/(fastPeriod+1);
	
	ema1 = np.zeros(close.size);
	ema2 = np.copy(ema1);
	ema3 = np.copy(ema1);
	ema4 = np.copy(ema1);
	
	for i in range(1,close.size):
		ema1[i] = k1 * (momentumVector[i]-ema1[i-1]) + ema1[i-1];
		ema2[i] = k2 * (ema1[i]-ema2[i-1])   + ema2[i-1];
		ema3[i] = k1 * (absMomentumVector[i]-ema3[i-1]) + ema3[i-1];
		ema4[i] = k2 * (ema3[i]-ema4[i-1])   + ema4[i-1];

	tsi = 100 * np.divide(ema2, ema4)
	return tsi;

def HHLL(high, low, periods=20):
	vectorSize = high.shape[0];
	lowestLowVector = np.zeros(vectorSize);
	lowestLowVector[0:periods] = min(low[0:periods]);
	for i in range((periods-1), vectorSize):
		lowestLowVector[i] = min(low[(i-periods+1):i+1]);
	highestHighVector = np.zeros(vectorSize);
	highestHighVector[0:periods] = max(high[0:periods]);
	for i in range((periods-1), vectorSize):
		highestHighVector[i] = max(high[(i-periods+1):i+1]);
	
	midPointVector = (highestHighVector + lowestLowVector) / 2;
	return np.hstack((highestHighVector.reshape((highestHighVector.size, 1)), lowestLowVector.reshape((lowestLowVector.size, 1)),
					 midPointVector.reshape((midPointVector.size, 1))))
	
def CMF(high, low, close, volume, periods=20):
	vectorSize = high.shape[0]; 
	moneyFlowMultiplier = ((close - low) - (high - close)).reshape((vectorSize,1)) * np.linalg.pinv((high - low).reshape((vectorSize,1)));
	moneyFlowVolume = np.dot(moneyFlowMultiplier,volume.reshape((vectorSize,1)));
	cmf = np.zeros(vectorSize);
	cmf[:] = np.NAN;
	for i in range((periods-1),vectorSize):
		cmf[i] = sum(moneyFlowVolume[i-periods+1:i+1]) / sum(volume[i-periods+1:i+1]);
	return cmf;

def FORCE(close, volume, periods=13):
	vectorSize = close.shape[0];
	force = np.zeros(vectorSize);
	force[0] = np.nan;
	force[1:] = (close[1:] - close[0:vectorSize-1]) * volume[1:];
	force[1:] = ta.EMA(force[1:], periods);
	return force;

def VR(high, low, close, periods=14):
	vectorSize = close.shape[0];
	highLowDiff = high - low;
	highCloseDiff = np.zeros(vectorSize);
	highCloseDiff[1:] = np.abs(high[1:]-close[0:vectorSize-1]);
	lowCloseDiff = np.zeros(vectorSize);
	lowCloseDiff[1:] = np.abs(low[1:]-close[0:vectorSize-1]);
	vectorsStacked = np.vstack((highLowDiff, highCloseDiff, lowCloseDiff));
	tr = np.amax(vectorsStacked, axis=0);
	vr = tr / ta.EMA(tr, periods);
	return vr;