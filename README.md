
This code is the official implementation for the IAMAHA presentation on [the EIDA project](https://eida.hypotheses.org/). Our project webpage can be found [here](https://imagine.enpc.fr/~kallelis/iamaha2023/).

We present two distinct approaches to address the task of vectorizing diagrams: a traditional method that leverages contour extraction and robust estimation techniques, and a modern, learning-based method that builds on the Line Segment Detector Transformer [LETR](https://arxiv.org/abs/2101.01909).


## Abstract 
The EIDA project explores the historical use of astronomical diagrams across Asia, Africa, and Europe. We aim to develop automatic image analysis tools to analyze and edit these diagrams without human annotation, gaining a refined understanding of their role in shaping and transmitting astronomy. In this paper, we present a method to detects lines and circles in historical diagrams, based on text removal, edge detection and RANSAC. We plan to compare this strong baseline with deep approaches. This work contributes to historical diagram vectorization, enabling novel methods of comparison and clustering, and offering fresh insights into the vast corpus of astronomical diagrams.

## Dataset

Our manually annotated sample dataset of 15 diagrams can be found [here](https://drive.google.com/drive/folders/1V0PEsLhMXmQYkgFlqEIAopCQNxvmOnuz?usp=drive_link). 


## Method

Detailed instructions for each approach can be found in their respective README.md files.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{kalleli2023eida, 
    title={{EIDA: Editing and analysing historical astronomical diagrams with artificial intelligence}}, 
    author={Kalleli, Syrine and Trigg, Scott and Albouy, Ségolène and Guessner, Samuel and Husson, Mathieu and Aubry, Mathieu}, 
    booktitle={IAMAHA}, 
    year={2023}
}