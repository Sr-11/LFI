I want to (hopefully) show something like with complete knowledge of train (X, Y) data and test (Z) data, 
our kernel maximizing MMD_LFI-var(MMD_LFI) outperforms MMD_TST-var(TST) and thus is preferred.

And perhaps we can say something like:
hey, training a different kernel for each test Z is too expensive so we only 
do part of our experiment on these stats and change focus onto TST kernel LFI 
(where you do not see Z during training).
which is convenient and factually true