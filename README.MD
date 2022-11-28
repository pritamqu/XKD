<h1 align="center"> 
XKD: Cross-modal Knowledge Distillation with Domain Alignment for Video Representation Learning
</h1>

<h3 align="center">
Under Review.
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
- [ ] Pretrained model weights
- [ ] Evaluation codes
- [ ] Training codes

#### ** Please check the project website for more details. The codes will be released soon. You may follow this repo to receive future updates. **


![XKD](/docs/assets/images/xkd.png?raw=true "XKD")


### Abstract
We present **XKD**, a novel self-supervised framework to learn meaningful representations from unlabelled video clips. XKD is trained with two pseudo tasks. First, masked data reconstruction is performed to learn modality-specific representations. Next, self-supervised cross-modal knowledge distillation is performed between the two modalities through teacher-student setups to learn complementary information. To identify the most effective information to transfer and also to tackle the domain gap between audio and visual modalities which could hinder knowledge transfer, we introduce a domain alignment strategy for effective cross-modal distillation. Lastly, to develop a general-purpose solution capable of handling both audio and visual streams, a modality-agnostic variant of our proposed framework is introduced, which uses the same backbone for both audio and visual modalities. Our proposed cross-modal knowledge distillation improves linear evaluation top-1 accuracy of video action classification by 8.4% on UCF101, 8.1% on HMDB51, 13.8% on Kinetics-Sound, and 14.2% on Kinetics400. Additionally, our modality-agnostic variant shows promising results in developing a general-purpose network capable of handling different data streams.



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
