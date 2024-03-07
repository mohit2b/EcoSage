# An EcoSage Assistant: Towards Building A Multimodal Plant Care Dialogue Assistant

This reporsitory contains the code for our ECIR 2024 paper "[An EcoSage Assistant: Towards Building A Multimodal Plant Care Dialogue Assistant](https://arxiv.org/abs/2401.06807)", Advances in Information Retrieval, 46th European Conference on Information Retrieval, ECIR 2024. 

An example dialogue from our dataset is illustrated in the figure below:

<img src="/image/dataset.png" alt="Dataset Example" width="700"/>
<!-- ![Dataset Example](/image/dataset.png) -->



>In recent times, there has been an increasing awareness about imminent environmental challenges, resulting in people showing a stronger dedication to taking care of the environment and nurturing green life. The current $19.6 billion indoor gardening industry, reflective of this growing sentiment, not only signifies a monetary value but also speaks of a profound human desire to reconnect with the natural world. However, several recent surveys cast a revealing light on the fate of plants within our care, with more than half succumbing primarily due to the silent menace of improper care. Thus, the need for accessible expertise capable of assisting and guiding individuals through the intricacies of plant care has become paramount more than ever. In this work, we make the very first attempt at building a plant care assistant, which aims to assist people with plant(-ing) concerns through conversations. We propose a plant care conversational dataset named Plantational, which contains around 1K dialogues between users and plant care experts. Our end-to-end proposed approach is two-fold : (i) We first benchmark the dataset with the help of various large language models (LLMs) and visual language model (VLM) by studying the impact of instruction tuning (zero-shot and few-shot prompting) and fine-tuning techniques on this task; (ii) finally, we build EcoSage, a multi-modal plant care assisting dialogue generation framework, incorporating an adapter-based modality infusion using a gated mechanism. We performed an extensive examination (both automated and manual evaluation) of the performance exhibited by various LLMs and VLM in the generation of the domain-specific dialogue responses to underscore the respective strengths and weaknesses of these diverse models.


* **Authors:** Mohit Tomar, Abhisek Tiwari, Tulika Saha, Prince Jha, Sriparna Saha

If you consider this work to be useful, please cite it as

```bash
@article{tomar2024ecosage,
  title={An EcoSage Assistant: Towards Building A Multimodal Plant Care Dialogue Assistant},
  author={Tomar, Mohit and Tiwari, Abhisek and Saha, Tulika and Jha, Prince and Saha, Sriparna},
  journal={arXiv preprint arXiv:2401.06807},
  year={2024}
}
```
This code has been adapted from the "[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)" paper.

# Contact

For any queries, feel free to contact Mohit Tomar (mohitsinghtomar9797@gmail.com)
