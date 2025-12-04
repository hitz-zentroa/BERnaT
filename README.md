# BERnaT: Basque Encoders for Representing Natural Textual Diversity

Submitted to LREC 2026

## Abstract

Language models depend on massive text corpora that are often filtered for quality, a process that can unintentionally
exclude non-standard linguistic varieties, reduce model robustness and reinforce representational biases. In this
paper, we argue that language models should aim to capture the full spectrum of language variation (dialectal,
historical, informal, etc.) rather than relying solely on standardized text. Focusing on Basque, a morphologically rich
and low-resource language, we construct new corpora combining standard, social media, and historical sources, and
pre-train the BERnaT family of encoder-only models in three configurations: standard, diverse, and combined. We
further propose an evaluation framework that separates Natural Language Understanding (NLU) tasks into standard
and diverse subsets to assess linguistic generalization. Results show that models trained on both standard and
diverse data consistently outperform those trained on standard corpora, improving performance across all task types
without compromising standard benchmark accuracy. These findings highlight the importance of linguistic diversity in
building inclusive, generalizable language models.

## Results

|                     | **AVG standard tasks** | **AVG diverse tasks** | **AVG overall** |
|---------------------|:----------------------:|:---------------------:|:---------------:|
| **BERnaT_standard** |                        |                       |                 |
| medium              |          74.10         |         70.30         |      72.58      |
| base                |          75.33         |         71.26         |      73.70      |
| large               |          76.83         |         73.13         |      75.35      |
| **BERnaT_diverse**  |                        |                       |                 |
| medium              |          71.66         |         69.91         |      70.96      |
| base                |          72.44         |         71.43         |      72.04      |
| large               |          74.48         |         71.87         |      73.43      |
| **BERnaT**          |                        |                       |                 |
| medium              |          73.56         |         70.59         |      72.37      |
| base                |          75.42         |         71.28         |      73.76      |
| large               |        **77.88**       |       **73.77**       |    **76.24**    |

## Acknowledgments

This work has been partially supported by the Basque Government (Research group funding IT1570-22 and IKER-GAITU project), the Spanish Ministry for Digital Transformation and Civil Service, and the EU-funded NextGenerationEU Recovery, Transformation and Resilience Plan (ILENIA project, 2022/TL22/00215335; and ALIA project). The project also received funding from the European Union’s Horizon Europe research and innovation program under Grant Agreement No 101135724, Topic HORIZON-CL4-2023-HUMAN-01-21 and DeepKnowledge (PID2021-127777OB-C21) founded by MCIN/AEI/10.13039/501100011033 and FEDER. Jaione Bengoetxea, Julen Etxaniz and Ekhi Azurmendi hold a PhD grant from the Basque Government (PRE_2024_1_0028, PRE_2024_2_0028 and PRE_2024_1_0035, respectively). Maite Heredia and Mikel Zubillaga hold a PhD grant from the University of the Basque Country UPV/EHU (PIF23/218 and PIF24/04, respectively). The models were trained on the Leonardo supercomputer at CINECA under the EuroHPC Joint Undertaking, project EHPC-EXT-2024E01-042.

## Citation:

To cite our work, please use:

```bibtex
@misc{azurmendi2025bernatbasqueencodersrepresenting,
      title={BERnaT: Basque Encoders for Representing Natural Textual Diversity}, 
      author={Ekhi Azurmendi and Joseba Fernandez de Landa and Jaione Bengoetxea and Maite Heredia and Julen Etxaniz and Mikel Zubillaga and Ander Soraluze and Aitor Soroa},
      year={2025},
      eprint={2512.03903},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.03903}, 
}
```