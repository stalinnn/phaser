import csv
import re
import os

def recover_sota_logs():
    # Raw logs from our conversation history
    raw_data = """
mamba | Step 0 | Loss 10.9561 | Gate 0.0000
mamba | Step 10 | Loss 7.3286 | Gate 0.0000
mamba | Step 20 | Loss 6.9524 | Gate 0.0000
mamba | Step 30 | Loss 6.6413 | Gate 0.0000
mamba | Step 40 | Loss 6.5016 | Gate 0.0000
mamba | Step 50 | Loss 6.2958 | Gate 0.0000
mamba | Step 60 | Loss 6.1106 | Gate 0.0000
mamba | Step 70 | Loss 5.8992 | Gate 0.0000
mamba | Step 80 | Loss 5.7067 | Gate 0.0000
mamba | Step 90 | Loss 5.6372 | Gate 0.0000
mamba | Step 100 | Loss 5.5090 | Gate 0.0000
mamba | Step 110 | Loss 5.3743 | Gate 0.0000
mamba | Step 120 | Loss 5.1966 | Gate 0.0000
mamba | Step 130 | Loss 5.0863 | Gate 0.0000
mamba | Step 140 | Loss 5.0036 | Gate 0.0000
mamba | Step 150 | Loss 4.8410 | Gate 0.0000
mamba | Step 160 | Loss 4.7267 | Gate 0.0000
mamba | Step 170 | Loss 4.7004 | Gate 0.0000
mamba | Step 180 | Loss 4.4813 | Gate 0.0000
mamba | Step 190 | Loss 4.3885 | Gate 0.0000
mamba | Step 200 | Loss 4.2668 | Gate 0.0000
mamba | Step 210 | Loss 4.0507 | Gate 0.0000
mamba | Step 220 | Loss 4.0472 | Gate 0.0000
mamba | Step 230 | Loss 4.0049 | Gate 0.0000
mamba | Step 240 | Loss 3.8786 | Gate 0.0000
mamba | Step 250 | Loss 3.6608 | Gate 0.0000
mamba | Step 260 | Loss 3.6322 | Gate 0.0000
mamba | Step 270 | Loss 3.5360 | Gate 0.0000
mamba | Step 280 | Loss 3.3196 | Gate 0.0000
mamba | Step 290 | Loss 3.2306 | Gate 0.0000
mamba | Step 300 | Loss 2.8622 | Gate 0.0000
mamba | Step 310 | Loss 2.9375 | Gate 0.0000
mamba | Step 320 | Loss 2.9017 | Gate 0.0000
mamba | Step 330 | Loss 2.6009 | Gate 0.0000
mamba | Step 340 | Loss 2.4375 | Gate 0.0000
mamba | Step 350 | Loss 2.1645 | Gate 0.0000
mamba | Step 360 | Loss 2.1288 | Gate 0.0000
mamba | Step 370 | Loss 1.9555 | Gate 0.0000
mamba | Step 380 | Loss 1.7927 | Gate 0.0000
mamba | Step 390 | Loss 1.5241 | Gate 0.0000
mamba | Step 400 | Loss 1.5174 | Gate 0.0000
mamba | Step 410 | Loss 1.0993 | Gate 0.0000
mamba | Step 420 | Loss 1.1290 | Gate 0.0000
mamba | Step 430 | Loss 1.0052 | Gate 0.0000
mamba | Step 440 | Loss 0.9621 | Gate 0.0000
mamba | Step 450 | Loss 0.8248 | Gate 0.0000
mamba | Step 460 | Loss 0.7272 | Gate 0.0000
mamba | Step 470 | Loss 0.5595 | Gate 0.0000
mamba | Step 480 | Loss 0.4946 | Gate 0.0000
mamba | Step 490 | Loss 0.3883 | Gate 0.0000
mamba | Step 500 | Loss 0.3831 | Gate 0.0000
transformer | Step 0 | Loss 10.9452 | Gate 0.0000
transformer | Step 10 | Loss 7.6731 | Gate 0.0000
transformer | Step 20 | Loss 7.4400 | Gate 0.0000
transformer | Step 30 | Loss 7.3793 | Gate 0.0000
transformer | Step 40 | Loss 7.3198 | Gate 0.0000
transformer | Step 50 | Loss 7.3924 | Gate 0.0000
transformer | Step 60 | Loss 7.4022 | Gate 0.0000
transformer | Step 70 | Loss 7.3708 | Gate 0.0000
transformer | Step 80 | Loss 7.3331 | Gate 0.0000
transformer | Step 90 | Loss 7.4103 | Gate 0.0000
transformer | Step 100 | Loss 7.3740 | Gate 0.0000
transformer | Step 110 | Loss 7.3874 | Gate 0.0000
transformer | Step 120 | Loss 7.3499 | Gate 0.0000
transformer | Step 130 | Loss 7.3655 | Gate 0.0000
transformer | Step 140 | Loss 7.4326 | Gate 0.0000
transformer | Step 150 | Loss 7.4073 | Gate 0.0000
transformer | Step 160 | Loss 7.3723 | Gate 0.0000
transformer | Step 170 | Loss 7.3401 | Gate 0.0000
transformer | Step 180 | Loss 7.4152 | Gate 0.0000
transformer | Step 190 | Loss 7.3313 | Gate 0.0000
transformer | Step 200 | Loss 7.3296 | Gate 0.0000
transformer | Step 210 | Loss 7.3841 | Gate 0.0000
transformer | Step 220 | Loss 7.3826 | Gate 0.0000
transformer | Step 230 | Loss 7.3840 | Gate 0.0000
transformer | Step 240 | Loss 7.3936 | Gate 0.0000
transformer | Step 250 | Loss 7.3633 | Gate 0.0000
transformer | Step 260 | Loss 7.3992 | Gate 0.0000
transformer | Step 270 | Loss 7.4009 | Gate 0.0000
transformer | Step 280 | Loss 7.3377 | Gate 0.0000
transformer | Step 290 | Loss 7.4018 | Gate 0.0000
transformer | Step 300 | Loss 7.3793 | Gate 0.0000
transformer | Step 310 | Loss 7.3460 | Gate 0.0000
transformer | Step 320 | Loss 7.3604 | Gate 0.0000
transformer | Step 330 | Loss 7.3907 | Gate 0.0000
transformer | Step 340 | Loss 7.3736 | Gate 0.0000
transformer | Step 350 | Loss 7.3294 | Gate 0.0000
transformer | Step 360 | Loss 7.4147 | Gate 0.0000
transformer | Step 370 | Loss 7.3426 | Gate 0.0000
transformer | Step 380 | Loss 7.3425 | Gate 0.0000
transformer | Step 390 | Loss 7.4139 | Gate 0.0000
transformer | Step 400 | Loss 7.3944 | Gate 0.0000
transformer | Step 410 | Loss 7.3649 | Gate 0.0000
transformer | Step 420 | Loss 7.3982 | Gate 0.0000
transformer | Step 430 | Loss 7.3391 | Gate 0.0000
transformer | Step 440 | Loss 7.3449 | Gate 0.0000
transformer | Step 450 | Loss 7.3407 | Gate 0.0000
transformer | Step 460 | Loss 7.3627 | Gate 0.0000
transformer | Step 470 | Loss 7.3500 | Gate 0.0000
transformer | Step 480 | Loss 7.3686 | Gate 0.0000
transformer | Step 490 | Loss 7.4045 | Gate 0.0000
transformer | Step 500 | Loss 7.2965 | Gate 0.0000
tgn | Step 0 | Loss 10.9700 | Gate 0.5015
tgn | Step 10 | Loss 7.7550 | Gate 0.4822
tgn | Step 20 | Loss 7.4686 | Gate 0.3552
tgn | Step 30 | Loss 7.3448 | Gate 0.2047
tgn | Step 40 | Loss 7.3686 | Gate 0.0837
tgn | Step 50 | Loss 7.3799 | Gate 0.0611
tgn | Step 60 | Loss 7.3426 | Gate 0.0501
tgn | Step 70 | Loss 7.3953 | Gate 0.0435
tgn | Step 80 | Loss 7.3398 | Gate 0.0383
tgn | Step 90 | Loss 7.3874 | Gate 0.0338
tgn | Step 100 | Loss 7.3351 | Gate 0.0300
tgn | Step 110 | Loss 7.3900 | Gate 0.0271
tgn | Step 120 | Loss 7.3116 | Gate 0.0246
tgn | Step 130 | Loss 7.3790 | Gate 0.0228
tgn | Step 140 | Loss 7.4046 | Gate 0.0215
tgn | Step 150 | Loss 7.3256 | Gate 0.0203
tgn | Step 160 | Loss 7.3688 | Gate 0.0196
tgn | Step 170 | Loss 7.3635 | Gate 0.0188
tgn | Step 180 | Loss 7.3948 | Gate 0.0181
tgn | Step 190 | Loss 7.3561 | Gate 0.0174
tgn | Step 200 | Loss 7.3715 | Gate 0.0171
tgn | Step 210 | Loss 7.3616 | Gate 0.0166
tgn | Step 220 | Loss 7.3825 | Gate 0.0161
tgn | Step 230 | Loss 7.3501 | Gate 0.0155
tgn | Step 240 | Loss 7.3031 | Gate 0.0157
tgn | Step 250 | Loss 7.3519 | Gate 0.0156
tgn | Step 260 | Loss 7.3437 | Gate 0.0144
tgn | Step 270 | Loss 7.3166 | Gate 0.0139
tgn | Step 280 | Loss 7.3172 | Gate 0.0136
tgn | Step 290 | Loss 7.3336 | Gate 0.0134
tgn | Step 300 | Loss 7.2918 | Gate 0.0131
tgn | Step 310 | Loss 7.3410 | Gate 0.0129
tgn | Step 320 | Loss 7.2624 | Gate 0.0131
tgn | Step 330 | Loss 7.2312 | Gate 0.0123
tgn | Step 340 | Loss 7.2075 | Gate 0.0112
tgn | Step 350 | Loss 7.1901 | Gate 0.0110
tgn | Step 360 | Loss 7.1528 | Gate 0.0112
tgn | Step 370 | Loss 7.1227 | Gate 0.0110
tgn | Step 380 | Loss 7.0990 | Gate 0.0111
tgn | Step 390 | Loss 7.0986 | Gate 0.0108
tgn | Step 400 | Loss 7.0940 | Gate 0.0098
tgn | Step 410 | Loss 6.9848 | Gate 0.0108
tgn | Step 420 | Loss 7.0703 | Gate 0.0111
tgn | Step 430 | Loss 7.0095 | Gate 0.0103
tgn | Step 440 | Loss 6.9554 | Gate 0.0102
tgn | Step 450 | Loss 6.9400 | Gate 0.0100
tgn | Step 460 | Loss 6.9460 | Gate 0.0097
tgn | Step 470 | Loss 6.9315 | Gate 0.0090
tgn | Step 480 | Loss 6.9601 | Gate 0.0096
tgn | Step 490 | Loss 6.9233 | Gate 0.0081
tgn | Step 500 | Loss 6.8993 | Gate 0.0085
"""
    
    os.makedirs('result_sota_a800', exist_ok=True)
    
    for line in raw_data.strip().split('\n'):
        if not line: continue
        parts = line.split(' | ')
        model = parts[0]
        step = parts[1].split(' ')[1]
        loss = parts[2].split(' ')[1]
        gate = parts[3].split(' ')[1]
        
        with open(f'result_sota_a800/log_{model}.csv', 'a') as f:
            f.write(f"{step},{loss},{gate}\n")
            
    print("Recovered SOTA logs!")

if __name__ == "__main__":
    recover_sota_logs()
