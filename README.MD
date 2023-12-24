<h1 align="center"> 
XKD: Cross-modal Knowledge Distillation with Domain Alignment for Video Representation Learning
</h1>

<h3 align="center">
AAAI 2024
</h3>
<h3 align="center">
<a href="https://www.pritamsarkar.com">Pritam Sarkar</a>
&nbsp;
Ali Etemad
</h3>
<h3 align="center"> 
<a href="https://arxiv.org/pdf/2211.13929.pdf">[Paper]</a> <!-- change with pub link -->
<!-- <a href="./docs/assets/files/xkd_supp.pdf">[Appendix]</a> 
 <a href="https://arxiv.org/pdf/2211.00000.pdf"> [ArXiv]</a>   -->
 <a href="https://github.com/pritamqu/XKD/"> [Code]</a> <a href="https://pritamqu.github.io/XKD/"> [Website]</a>
</h3>


### Updates
- [x] Paper
- [x] Pretrained model weights
- [ ] Evaluation codes
- [x] Training codes
- [ ] More documentations
- [x] [Supplementary material](/docs/assets/xkd_supp.pdf)

#### ** Please check the project website for more details. The codes will be released soon. You may follow this repo to receive future updates. **


![XKD](/docs/assets/images/xkd.png?raw=true "XKD")


### Abstract
We present XKD, a novel self-supervised framework to learn meaningful representations from unlabelled videos. XKD is trained with two pseudo objectives. First, masked data reconstruction is performed to learn modality-specific representations from audio and visual streams. Next, self-supervised cross-modal knowledge distillation is performed between the two modalities through a teacher-student setup to learn complementary information. We introduce a novel domain alignment strategy to tackle domain discrepancy between audio and visual modalities enabling effective cross-modal knowledge distillation.
Additionally, to develop a general-purpose network capable of handling both audio and visual streams, modality-agnostic variants of XKD are introduced, which use the same pretrained backbone for different audio and visual tasks. Our proposed cross-modal knowledge distillation improves video action classification by 8% to 14% on UCF101, HMDB51, and Kinetics400. Additionally, XKD improves multimodal action classification by 5.5% on Kinetics-Sound. XKD shows state-of-the-art performance in sound classification on ESC50, achieving top-1 accuracy of 96.5%.



### Citation
If you find this repository useful, please consider giving a star :star: and citation using the given BibTeX entry:

```
@misc{sarkar2022xkd,
      title={XKD: Cross-modal Knowledge Distillation with Domain Alignment for Video Representation Learning}, 
      author={Pritam Sarkar and Ali Etemad},
      year={2022},
      eprint={2211.13929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgments
We are grateful to **Bank of Montreal** and **Mitacs** for funding this research. We are also thankful to **SciNet HPC Consortium** for helping with the computation resources.

### Question
You may directly contact me at <pritam.sarkar@queensu.ca> or connect with me on [LinkedIn](https://www.linkedin.com/in/sarkarpritam/).
