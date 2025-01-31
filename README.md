# VisioSLR-Sign-Language-Video-Recognition-Framework
VisioSLR provides a precise measurement of translating signs for developing an end-to-end computational translation system. 

## Abstract

With the emergence of AI for good, there has been an increasing interest in building computer vision data-driven deep learning inclusive AI solutions. Sign language Recognition (SLR) has gained attention recently. It is an essential component of a sign-to-text translation system to support the deaf and hard-of-hearing population. This paper presents a computer VISIOn data-driven deep learning framework for Sign Language video Recognition (VisoSLR). VisioSLR provides a precise measurement of translating signs for developing an end-to-end computational translation system. Considering the scarcity of sign language datasets, which hinders the development of an accurate recognition model, we evaluate the performance of our framework by fine-tuning the very well-known YOLO models, which are built from a signs-unrelated collection of images and videos, using a small-sized sign language dataset. Gathering a sign language dataset for signs training would involve an enormous amount of time to collect and annotate videos in different environmental setups and multiple signers, in addition to the training time of a model. Numerical evaluations of VisioSLR show that our framework recognizes signs with a mean average precision of 97.4%, 97.1%, and 95.5% and 11, 12, and 12 milliseconds of recognition time on YOLOv8m, YOLOv9m, and YOLOv11m, respectively.

## About this work

The major contributions of this work are as follows:

- At the leading edge, we propose an innovative sign language recognition framework that harnesses the power of fine-tuning, YOLO models, OpenCV, MediaPipe, and NumPy. This framework is designed to enable end-to-end implementation of real-time sign language  recognition application.
- We conduct a comparative analysis of fine-tuning YOLOv8, v9, and v11 pre-trained models using a Roboflow ASL sign language small dataset.
- We deploy the most accurate YOLO model developed using VisioSLR in a real-time SLR application, demonstrating our VisioSLR practical feasibility. 

## Cite this work

Leila Ismail, Nada Shahin, Henod Tesfaye, and Alain Hennebelle. "VisioSLR: A Vision Data-Driven Framework for Sign Language Video Recognition and Performance Evaluation on Fine-Tuned YOLO Models." The 16th International Conference on Ambient Systems, Networks, and Technologies April 22-24, 2025, Patras, Greece.

## References

Nada Shahin, and Leila Ismail."From rule-based models to deep learning transformers architectures for natural language processing and sign language translation systems: survey, taxonomy and performance evaluation." Artificial Intelligence Review. 2024.57:271. https://doi.org/10.1007/s10462-024-10895-z.

Nada Shahin and Leila Ismail. "GLoT: A Novel Gated-Logarithmic Transformer for Efficient Sign Language Translation." IEEE Future Networks World Forum. 2024.

Nada Shahin and Leila Ismail. "ChatGPT, Let us Chat Sign Language: Experiments, Architectural Elements, Challenges and Research Directions." The International Symposium on Networks, Computers and Communications. ISNCC.  2023.
