# Deep Learning Lecture Notes and Experiments

## 2022 Version (Latest)
Welcome to the 2022 version of Deep Learning course. We made major changes in the coverage and delivery of this course to reflect the recent advances in the field.

### Install
Assuming you already have  `anaconda` or `venv`, install the required python packages to run the experiments in this version.

`pip install -r requirements.txt `

### Coverage:

| **AI, ML and Deep Learning** | | | |
| :--- | :---: | :---: | :---: |
| &nbsp;&nbsp;&nbsp;&nbsp;Overview | [PDF](versions/2022/overview/Overview.pdf) | [YouTube](https://youtu.be/zU37kvvkz0o) | -  |
| **Toolkit**| | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Development Environment<br> &nbsp;&nbsp;&nbsp;&nbsp;and Code Editor | [PDF](versions/2022/tools/Toolkit_Env_Editor.pdf) | [YouTube](https://youtu.be/LildU3tGGEo) | -  |
| &nbsp;&nbsp;&nbsp;&nbsp;Python | [PDF](versions/2022/tools/Toolkit_Python.pdf)| [YouTube](https://youtu.be/4Q1G5GuIXw8) | -  |
| &nbsp;&nbsp;&nbsp;&nbsp;Numpy | [PDF](versions/2022/tools/Toolkit_Numpy.pdf) | [YouTube](https://youtu.be/_E9dnUY1Ets) | [Jupyter](versions/2022/tools/python/np_demo.ipynb) |
| &nbsp;&nbsp;&nbsp;&nbsp;Einsum | [PDF](versions/2022/tools/Toolkit_Einsum.pdf) | [YouTube](https://youtu.be/IUs7aWs-axM) | [Jupyter](versions/2022/tools/python/einsum_demo.ipynb) |
| &nbsp;&nbsp;&nbsp;&nbsp;Einops | [PDF](versions/2022/tools/Toolkit_Einops.pdf) | [YouTube](https://youtu.be/ll1BlfYd4mU) | [Jupyter](versions/2022/tools/python/einops_demo.ipynb) |
| &nbsp;&nbsp;&nbsp;&nbsp;PyTorch & Timm | [PDF](versions/2022/tools/Toolkit_PyTorch.pdf) | [YouTube](https://youtu.be/mK0CHqLCoXA) | [PyTorch/Timm](versions/2022/tools/python/pytorch_demo.ipynb) & <br> [Input](versions/2022/tools/python/input_demo.ipynb) Jupyter|
| &nbsp;&nbsp;&nbsp;&nbsp;Gradio & Hugging Face | [PDF](versions/2022/tools/Toolkit_Gradio.pdf) | [YouTube](https://youtu.be/b1NgUiTIUMc) | [Jupyter](versions/2022/tools/python/gradio_demo.ipynb) |
| &nbsp;&nbsp;&nbsp;&nbsp;Weights and Biases|  | | [Jupyter](versions/2022/tools/python/wandb_demo.ipynb) |
| **Datasets** | Soon | | |
| **Supervised Learning** | Soon | | |
| **Building blocks:<br> MLPs, CNNs, RNNs, Transformers** | Soon | | |
| **Backpropagation** | Soon | | |
| **Optimization** | Soon | | |
| **Regularization** | Soon | | |
| **Unsupervised Learning** | Soon | | |
| **AutoEncoders** | Soon | | |
| **Variational AutoEncoders** | Soon | | |
| **Practical Applications:<br>Vision, Speech, NLP** | Soon | | |


### What is new in 2022 version:

1) Emphasis on tools to use and deploy deep learning models. In the past, we learn how to build and train models to perform certain tasks. However, often times we want to use a pre-trained model for immediate deployment. testing or demonstration. Hence, we will use tools such as `huggingface`, `gradio` and `streamlit` in our discussions.

2) Emphasis on understanding deep learning building blocks. The ability to build, train and test models is important. However, when we want to optimize and deploy a deep learning model on a new hardware or run it on production, we need an in-depth understanding of the code implementation of our algorithms. Hence, there will be emphasis on low-level algorithms and their code implementations.

3) Emphasis on practical applications. Deep learning can do a lot more than recognition. Hence, we will highlight practical applications in vision (detection, segmentation), speech (ASR, TTS) and text (sentiment, summarization).

4) Various levels of abstraction. We will present deep learning concepts from low-level `numpy` and `einops`, to mid-level framework such as PyTorch, and to high-level APIs such as `huggingface`, `gradio` and `streamlit`. This enables us to use deep learning principles depending on the problem constraints.

5) Emphasis on individual presentation of assignments, machine exercises and projects. Online learning is hard. To maximize student learning, this course focuses on exchange of ideas to ensure individual student progress. 


### Star, Fork, Cite
If you find this work useful, please give it a star, fork, or cite:

```
@misc{atienza2020dl,
  title={Deep Learning Lecture Notes},
  author={Atienza, Rowel},
  year={2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/roatienza/Deep-Learning-Experiments}},
}
```



