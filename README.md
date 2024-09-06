# NavHint: Vision and Language Navigation Agent with a Hint Generator
This repo provides the official implementation of our [Navhint](https://arxiv.org/abs/2402.02559) (EACL2024 Findings)

> Abstract: Existing work on vision and language navigation mainly relies on navigation-related losses to establish the connection between vision and language modalities, neglecting aspects of helping the navigation agent build a deep understanding of the visual environment.
In our work, we provide indirect supervision to the navigation agent through a hint generator that provides detailed visual descriptions.
The hint generator assists the navigation agent in developing a global understanding of the visual environment. It directs the agent's attention toward related navigation details, including the relevant sub-instruction, potential challenges in recognition and ambiguities in grounding, and the targeted viewpoint description. 
To train the hint generator, we construct a synthetic dataset based on landmarks in the instructions and visible and distinctive objects in the visual environment.
We evaluate our method on the R2R and R4R datasets and achieve state-of-the-art on several metrics. 
The experimental results demonstrate that generating hints not only enhances the navigation performance but also helps improve the interpretability of the agent's actions.

### Framework
![](navhint.png)
