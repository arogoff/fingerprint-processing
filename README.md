The purpose of this code is to perform biometric fingerprint image processing and feature extraction for authentication decisions.

Using the NIST 8-Bit Gray Scale Images of Fingerprint Image Groups (FIGS) database, we break it up into two sets:
a. TRAIN: first 15000 image pairs (f0001-f1499 & s0001-s1499)
b. TEST: last 500 image pairs (f1501-f2000 & s1501-s2000)

Three methods are used:
1. Bifurcation Analysis
2. Crossover Analysis
3. Dot/Island Analysis

Additionally, one hybrid method is created, using all three methods listed above.
