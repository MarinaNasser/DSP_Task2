# Equalizer

Systems and biomedical engineering, Cairo University.
Level 3, 1st semester.
### Course: Digital signal processing, Task 1 (11/10/2022)

### Members
| Team Members' Names                                  | Section| B.N. |
|------------------------------------------------------|:------:|:----:|
| [Marina Nasser](https://github.com/MarinaNasser)     |    2   |  12  |
| [Ghofran Mohamed](https://github.com/GuFranMohammed) |    2   |  8   |
| [Nourhan Sayed](https://github.com/nourhansayed102)  |    2   |  41  |
| [Mayar Ehab](https://github.com/mayarehab)           |    2   |  47  |

## Main Idea
### Signal processing, by applying FFT to a signal, control the amplitude of the frequncies within the signal and displaying it after applying Inverse Fourier Transform. 


## Fast Fourier Transform (FFT)
Fast Fourier Transformation(FFT) is a mathematical algorithm that calculates Discrete Fourier Transform(DFT) of a given sequence. The only difference between FT(Fourier Transform) and FFT is that FT considers a continuous signal while FFT takes a discrete signal as input. DFT converts a sequence (discrete signal) into its frequency constituents just like FT does for a continuous signal. In our case, we have a sequence of amplitudes that were sampled from a continuous audio signal. DFT or FFT algorithm can convert this time-domain discrete signal into a frequency-domain.
   
   
![1_152qTVoPawbVtWPJtGeqWg](https://user-images.githubusercontent.com/81776523/200606602-359c841c-05eb-493a-b3b6-52e5ca1ab1f5.png)

## Features
* Uploading Csv or Wav type file.
* Contains 4 modes ("Uniform Range Mode", "Vowels Mode", "Musical Instruments Mode", "Biological Signal Abnormalities").
* Displaying orignal and modified audio in vowels and musical insturments modes.
* Displaying static and dynamic plots for the orignal and modified signal.
* A spectorogram for both orignal and modified signals.

## Modes Description 
* "Uniform Range Mode" : the total frequency range of the input signal is divided uniformly into 10 equal ranges of frequencies, each is controlled by one slider.
* "Vowels Mode" : each slider can control the magnitude of specific vowel.
* "Musical Instruments Mode" : each slider can control the magnitude of a specific musical instrument.
* "Biological Signal Abnormalities" : one slider control the normal ECG signal and the other controls the arrhythmia within the signal.

## Spectrogram
Visual representation of frequencies of a given signal with time is called Spectrogram. In a spectrogram representation plot — one axis represents the time, the second axis represents frequencies and the colors represent magnitude (amplitude) of the observed frequency at a particular time.

## Demos 
### Uniform Range Mode
![Equalizer - Google Chrome 11_19_2022 2_06_10 AM](https://user-images.githubusercontent.com/93389441/202823549-658d912a-8a31-4ab1-ace7-9fada328f1fc.png)
 
### Vowels Mode
![Equalizer - Google Chrome 11_19_2022 2_07_19 AM](https://user-images.githubusercontent.com/93389441/202823556-a261a0d2-48e3-4dd3-a27a-434720991a2b.png)

### Spectrogram and Dynamic plotting in Musical Instruments Mode
![Equalizer - Google Chrome 11_19_2022 2_08_29 AM](https://user-images.githubusercontent.com/93389441/202823560-1db0715a-6113-44ed-ae5d-960be2ecf7c0.png)

### Biological Signal Abnormalities 
![Equalizer - Google Chrome 11_19_2022 2_09_25 AM](https://user-images.githubusercontent.com/93389441/202823567-443d96c4-a5dd-40cb-84ab-fa54831b8f74.png)

