import unittest
from indicators import TalibExtension
import numpy as np;

class TalibExtensionTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.infoMatrix = np.genfromtxt('exampleTAMatrix.csv', delimiter=',');
        cls.high = cls.infoMatrix[:, 0];
        cls.low = cls.infoMatrix[:, 1];
        cls.close = cls.infoMatrix[:, 2];
        cls.volume = cls.infoMatrix[:, 3];
        
    def testKdj(self):
        kdjMatrix = TalibExtension.KDJ(self.high, self.low, self.close, 5, 3);
        self.assertEqual(round(kdjMatrix[0,0], 5), 45.92033, 'Error in kdj computation');
        self.assertEqual(True, np.isnan(kdjMatrix[0,1]), 'Error in kdj computation');
        self.assertEqual(True, np.isnan(kdjMatrix[1,2]), 'Error in kdj computation');
        self.assertEqual(round(kdjMatrix[2,2],5), 137.14009, 'Error in kdj computation');
        self.assertEqual(round(kdjMatrix[10,2],5), -27.25001, 'Error in kdj computation');
        self.assertEqual(round(kdjMatrix[13,0],5), 13.93087, 'Error in kdj computation');
        self.assertEqual(round(kdjMatrix[19,2],5), -14.40028, 'Error in kdj computation');

    def testTsi(self):
        tsiMatrix = TalibExtension.TSI(self.close, 5, 3);
        self.assertEqual(True, np.isnan(tsiMatrix[0]), 'Error in tsi computation');
        self.assertEqual(round(tsiMatrix[1],5), -100, 'Error in tsi computation');
        self.assertEqual(round(tsiMatrix[2],5), 70.39519, 'Error in tsi computation');
        self.assertEqual(round(tsiMatrix[5],5), -7.17987, 'Error in tsi computation');
        self.assertEqual(round(tsiMatrix[10],5), -19.06228, 'Error in tsi computation');
        self.assertEqual(round(tsiMatrix[19],5), -11.93875, 'Error in tsi computation');
        
    def testHhll(self):
        hhllMatrix = TalibExtension.HHLL(self.high, self.low, 5);
        self.assertEqual(hhllMatrix[0,0], 9.1338, 'Error in hhll computation');
        self.assertEqual(hhllMatrix[5,1], 0.35712, 'Error in hhll computation');
        self.assertEqual(hhllMatrix[15,2], 5.012115, 'Error in hhll computation');
        self.assertEqual(hhllMatrix[19,2], 4.96968, 'Error in hhll computation');
        
    def testCmf(self):
        cmfMatrix = TalibExtension.CMF(self.high, self.low, self.close, self.volume, 6);
        self.assertEqual(True, np.isnan(cmfMatrix[0]), 'Error in cmf computation');
        self.assertEqual(True, np.isnan(cmfMatrix[4]), 'Error in cmf computation');
        self.assertEqual(round(cmfMatrix[6],5), -0.07716, 'Error in cmf computation');
        self.assertEqual(round(cmfMatrix[19],5), -0.14178, 'Error in cmf computation');
        
    def testForce(self):
        forceMatrix = TalibExtension.FORCE(self.close, self.volume, 6);
        print(forceMatrix);
        self.assertEqual(True, np.isnan(forceMatrix[0]), 'Error in force computation');
        self.assertEqual(True, np.isnan(forceMatrix[5]), 'Error in force computation');
        self.assertEqual(round(forceMatrix[6],5), -482.06502, 'Error in force computation');
        self.assertEqual(round(forceMatrix[19],5), -3219.86027, 'Error in force computation');
        
    def testVr(self):
        vrMatrix = TalibExtension.VR(self.high, self.low, self.close, 6);
        print(vrMatrix);