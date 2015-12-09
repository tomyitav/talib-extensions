# talib-extensions
Implementing technical indicators that are not implemented in ta-lib.
usage is very similar to ta-lib, for example, compute Highest high, lowest low indicator: (extracted from test)

from indicators import TalibExtension

hhllMatrix = TalibExtension.HHLL(self.high, self.low, 5);

For other examples - see tests package
